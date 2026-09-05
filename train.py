import copy
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from scipy.linalg import hadamard

from utils.experiment import state_dict_sha256
from utils.tools import CalcTopMapPerQuery, CalcTopMapWithPR, compute_result

torch.multiprocessing.set_sharing_strategy("file_system")


class CSQLoss(torch.nn.Module):
    def __init__(self, args, bit, hash_center):
        super().__init__()
        self.hash_targets = hash_center.to(args.device)
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(args.device)
        self.criterion = torch.nn.BCELoss().to(args.device)

    def forward(self, output, labels, indices, args):
        del indices
        continuous_code = output.tanh()
        target_center = self.label2center(labels)
        center_loss = self.criterion(
            0.5 * (continuous_code + 1), 0.5 * (target_center + 1)
        )
        quantization_loss = (continuous_code.abs() - 1).pow(2).mean()
        return center_loss + args.lambd * quantization_loss

    def label2center(self, labels):
        return self.hash_targets[labels.argmax(dim=1)]

    def get_hash_targets(self, n_class, bit):
        base = hadamard(bit)
        targets = torch.from_numpy(np.concatenate((base, -base), 0)[:n_class]).float()
        if 2 * bit < n_class:
            targets.resize_(n_class, bit)
            for _ in range(20):
                for index in range(2 * bit, n_class):
                    ones = torch.ones(bit)
                    ones[random.sample(list(range(bit)), bit // 2)] = -1
                    targets[index] = ones
                distances = np.asarray(
                    [
                        sum(targets[i] != targets[j])
                        for i in range(n_class)
                        for j in range(i + 1, n_class)
                    ]
                )
                if distances.min() > bit / 4 and distances.mean() >= bit / 2:
                    break
        return targets


def _evaluate(query_loader, database_loader, net, args, num_database):
    query_binary, query_label = compute_result(query_loader, net, args.device)
    database_binary, database_label = compute_result(
        database_loader, net, args.device
    )
    map_values, pr_data = CalcTopMapWithPR(
        query_binary.numpy(),
        query_label.numpy(),
        database_binary.numpy(),
        database_label.numpy(),
        args.topK,
        num_database,
    )
    metrics = {
        "mAP@ALL": float(map_values[0]),
        "mAP@100": float(map_values[1]),
        "mAP@1000": float(map_values[2]),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"Non-finite retrieval metrics: {metrics}")
    return metrics, pr_data, query_binary, query_label, database_binary, database_label


def _save_curves(run_dir, losses, learning_rates):
    figure = plt.figure()
    plt.plot(range(1, len(learning_rates) + 1), learning_rates)
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.title("Learning rate")
    figure.savefig(os.path.join(run_dir, "learning_rate.png"), dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure = plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label="loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    figure.savefig(os.path.join(run_dir, "training_loss.png"), dpi=160, bbox_inches="tight")
    plt.close(figure)


def validate_frozen_query_protocol(summary, args):
    if summary.get("query_evaluation_count", 0) != 0:
        raise RuntimeError(
            "This frozen run has already been evaluated on the query set; "
            "a repeated evaluation is rejected"
        )
    expected_protocol = {
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
        "split_sha256": args.split_hash,
        "code_length": args.code_length,
        "topK": args.topK,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected_protocol.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Frozen query protocol mismatch: {mismatches}")

def train_val(
    args,
    hash_center,
    train_loader,
    val_loader,
    database_loader,
    num_database,
):
    print(f"========== start SHC {args.method} network training ==========")
    existing_metrics_path = os.path.join(args.run_dir, "metrics.json")
    if os.path.exists(existing_metrics_path):
        with open(existing_metrics_path, "r", encoding="utf-8") as f:
            existing_summary = json.load(f)
        if existing_summary.get("query_evaluation_count", 0) > 0:
            raise RuntimeError(
                "Refusing to overwrite a finalized run that has already accessed "
                "the query set; change the configuration to create a new run"
            )
    bit = args.code_length
    net = args.net(bit).to(args.device)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch, eta_min=1e-7
    )

    hash_center = hash_center.float()
    if hash_center.shape != (args.num_classes, bit):
        hash_center = hash_center.T
    criterion = CSQLoss(args, bit, hash_center)

    best_validation_map = float("-inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu() for key, value in net.state_dict().items()}
    history, losses, learning_rates = [], [], []

    for epoch in range(args.epoch):
        started = time.time()
        learning_rates.append(optimizer.param_groups[0]["lr"])
        net.train()
        total_loss = 0.0
        for images, labels, indices in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = net(images)
            loss = criterion(output, labels, indices, args)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite hash loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        epoch_loss = total_loss / max(len(train_loader), 1)
        losses.append(epoch_loss)

        should_validate = (epoch + 1) % args.test_map == 0 or epoch + 1 == args.epoch
        record = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "lr": learning_rates[-1],
            "seconds": time.time() - started,
        }
        if should_validate:
            validation_metrics, _, _, _, _, _ = _evaluate(
                val_loader, database_loader, net, args, num_database
            )
            record["validation"] = validation_metrics
            if validation_metrics["mAP@ALL"] > best_validation_map:
                best_validation_map = validation_metrics["mAP@ALL"]
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu() for key, value in net.state_dict().items()
                }
            print(
                f"{args.info} epoch={epoch + 1} loss={epoch_loss:.5f} "
                f"val_ALL={validation_metrics['mAP@ALL']:.5f} "
                f"val_100={validation_metrics['mAP@100']:.5f} "
                f"val_1000={validation_metrics['mAP@1000']:.5f}"
            )
        else:
            print(f"{args.info} epoch={epoch + 1} loss={epoch_loss:.5f}")
        history.append(record)

    net.load_state_dict(best_state)
    net.to(args.device)
    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
        "split_sha256": args.split_hash,
        "code_length": args.code_length,
        "topK": args.topK,
        "classifier_sha256": getattr(args, "classifier_sha256", None),
        "similarity_config_hash": getattr(args, "similarity_hash", None),
        "selected_epoch": best_epoch,
        "selection_metric": "validation mAP@ALL",
        "best_validation_mAP@ALL": best_validation_map,
        "best_model_sha256": state_dict_sha256(best_state),
        "query_metrics": None,
        "query_evaluation_count": 0,
    }

    os.makedirs(args.run_dir, exist_ok=True)
    torch.save(best_state, os.path.join(args.run_dir, "best_model_state.pt"))
    with open(os.path.join(args.run_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_curves(args.run_dir, losses, learning_rates)

    print(
        f"Selected epoch {best_epoch} by validation mAP@ALL="
        f"{best_validation_map:.5f}; query set was not evaluated"
    )
    print(f"========== finished SHC {args.method} network training ==========")
    return summary


def evaluate_query_checkpoint(
    args,
    checkpoint_path,
    query_loader,
    database_loader,
    num_database,
):
    """Evaluate a frozen checkpoint once and persist the auditable query outputs."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    run_dir = os.path.dirname(checkpoint_path)
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Missing training metrics beside checkpoint: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    validate_frozen_query_protocol(summary, args)

    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Invalid checkpoint state_dict: {checkpoint_path}")
    checkpoint_sha256 = state_dict_sha256(state)
    if checkpoint_sha256 != summary.get("best_model_sha256"):
        raise RuntimeError(
            "Checkpoint SHA256 does not match the frozen validation-selected model"
        )
    net = args.net(args.code_length).to(args.device)
    net.load_state_dict(state)
    metrics, pr_data, query_binary, query_label, database_binary, database_label = (
        _evaluate(query_loader, database_loader, net, args, num_database)
    )
    per_query_ap = CalcTopMapPerQuery(
        query_binary.numpy(),
        query_label.numpy(),
        database_binary.numpy(),
        database_label.numpy(),
        args.topK,
    )
    torch.save(
        {
            "query_binary": query_binary,
            "query_label": query_label,
            "database_binary": database_binary,
            "database_label": database_label,
            "pr_data": pr_data,
            "per_query_ap": per_query_ap,
        },
        os.path.join(run_dir, "retrieval_outputs.pt"),
    )
    summary["query_metrics"] = metrics
    summary["query_evaluation_count"] = 1
    summary["query_checkpoint_sha256"] = checkpoint_sha256
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"Final frozen query evaluation: mAP@ALL={metrics['mAP@ALL']:.5f}, "
        f"mAP@100={metrics['mAP@100']:.5f}, "
        f"mAP@1000={metrics['mAP@1000']:.5f}"
    )
    return summary
