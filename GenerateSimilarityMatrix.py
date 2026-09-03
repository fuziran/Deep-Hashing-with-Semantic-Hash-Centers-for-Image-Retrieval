import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from network import ClassifyNet
from utils.experiment import (
    build_cache_metadata,
    load_cache,
    save_cache,
    stable_config_hash,
)


def normalize_and_symmetrize_similarity(class_average, eps=1e-12):
    """Apply the paper order: row normalization, symmetrization, diagonal=1."""
    row_mean = class_average.mean(dim=1, keepdim=True)
    centered = class_average - row_mean
    scale = centered.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    normalized = centered / scale
    similarity = (normalized + normalized.T) / 2
    similarity.fill_diagonal_(1.0)
    return similarity


def _accuracy(net, dataloader, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            targets = labels.argmax(dim=1).to(device, non_blocking=True)
            logits, _ = net(images)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
    return correct / total if total else 0.0


def _classifier_cache(args):
    metadata = build_cache_metadata(args, "classifier")
    path = os.path.join(
        args.output_dir,
        "ClassificationNet",
        f"{args.dataset}_{metadata['config_hash']}.pt",
    )
    return path, metadata


def TrainClassificationNetwork(args, train_loader, val_loader):
    print("========== start classification network ==========")
    cache_path, metadata = _classifier_cache(args)
    if not args.force_recompute:
        cached_state = load_cache(cache_path, metadata)
        if cached_state is not None:
            net = ClassifyNet(args.num_classes).to(args.device)
            net.load_state_dict(cached_state)
            print(f"Loaded audited classifier cache: {cache_path}")
            return net

    net = ClassifyNet(args.num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.classify_epoch)
    best_accuracy = float("-inf")
    best_state = copy.deepcopy(net.state_dict())
    history = []

    for epoch in range(args.classify_epoch):
        started = time.time()
        net.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images = images.to(args.device, non_blocking=True)
            targets = labels.argmax(dim=1).to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = net(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite classification loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * targets.numel()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
        scheduler.step()

        should_validate = (
            (epoch + 1) % args.test_map == 0 or epoch + 1 == args.classify_epoch
        )
        val_accuracy = None
        if should_validate:
            val_accuracy = _accuracy(net, val_loader, args.device)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_state = copy.deepcopy(net.state_dict())
        record = {
            "epoch": epoch + 1,
            "loss": epoch_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "validation_accuracy": val_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(record)
        if should_validate:
            print(
                f"classifier {epoch + 1}/{args.classify_epoch} "
                f"loss={record['loss']:.5f} train={record['train_accuracy']:.4%} "
                f"validation={val_accuracy:.4%} best={best_accuracy:.4%}"
            )

    best_state = {key: value.detach().cpu() for key, value in best_state.items()}
    save_cache(cache_path, best_state, metadata)
    with open(
        os.path.join(args.run_dir, "classification_history.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    net.load_state_dict(best_state)
    net.to(args.device)
    print(f"Best classifier validation accuracy: {best_accuracy:.4%}")
    return net


def _similarity_cache(args):
    metadata = build_cache_metadata(args, "similarity")
    path = os.path.join(
        args.output_dir,
        "SimilarityMatrix",
        f"{args.dataset}_{metadata['config_hash']}.pt",
    )
    return path, metadata


def load_explicit_similarity_cache(args, path):
    """Load an audited Stage 1 artifact across a Stage 2-only code revision."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(("value", "metadata")) - set(payload):
        raise RuntimeError(f"Invalid explicit similarity cache: {path}")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Invalid similarity cache metadata: {path}")
    stored_without_hash = dict(metadata)
    stored_hash = stored_without_hash.pop("config_hash", None)
    if stored_hash != stable_config_hash(stored_without_hash):
        raise RuntimeError(f"Similarity cache metadata hash is invalid: {path}")

    expected = build_cache_metadata(args, "similarity")
    provenance_fields = {"git_commit", "config_hash"}
    mismatches = {
        key: (expected[key], metadata.get(key))
        for key in expected.keys() - provenance_fields
        if expected[key] != metadata.get(key)
    }
    if mismatches:
        raise RuntimeError(
            f"Explicit similarity cache is incompatible: {path}\n"
            f"mismatches={mismatches}"
        )

    similarity = payload["value"]
    expected_shape = (args.num_classes, args.num_classes)
    if not isinstance(similarity, torch.Tensor) or similarity.shape != expected_shape:
        raise RuntimeError(
            f"Similarity cache shape mismatch: expected={expected_shape}, "
            f"actual={getattr(similarity, 'shape', None)}"
        )
    if not torch.isfinite(similarity).all():
        raise RuntimeError("Explicit similarity cache contains NaN or Inf")
    if similarity.min() < -1.000001 or similarity.max() > 1.000001:
        raise RuntimeError("Explicit similarity cache values are outside [-1, 1]")
    if not torch.allclose(similarity, similarity.T, atol=1e-7, rtol=1e-6):
        raise RuntimeError("Explicit similarity cache is not symmetric")
    expected_diagonal = torch.ones(
        args.num_classes, dtype=similarity.dtype, device=similarity.device
    )
    if not torch.allclose(
        similarity.diag(), expected_diagonal, atol=1e-7, rtol=1e-6
    ):
        raise RuntimeError("Explicit similarity cache diagonal is not one")

    args.similarity_hash = stored_hash
    print(f"Loaded explicit audited similarity cache: {path}")
    print(
        "Similarity provenance: "
        f"git_commit={metadata.get('git_commit')} config_hash={stored_hash}"
    )
    return similarity


def GenerateSimilarityMatrix(args, train_loader, relation_loader, val_loader):
    if args.similarity_cache:
        return load_explicit_similarity_cache(args, args.similarity_cache)
    cache_path, metadata = _similarity_cache(args)
    args.similarity_hash = metadata["config_hash"]
    if not args.force_recompute:
        cached = load_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded audited similarity cache: {cache_path}")
            return cached

    net = TrainClassificationNetwork(args, train_loader, val_loader)
    print("========== start similarity matrix ==========")
    class_sum = torch.zeros(
        args.num_classes, args.num_classes, device=args.device, dtype=torch.float64
    )
    class_count = torch.zeros(
        args.num_classes, device=args.device, dtype=torch.float64
    )
    net.eval()
    with torch.no_grad():
        for images, labels, _ in relation_loader:
            images = images.to(args.device, non_blocking=True)
            targets = labels.argmax(dim=1).to(args.device, non_blocking=True)
            logits, _ = net(images)
            mask_targets = (
                logits.argmax(dim=1)
                if args.mask_strategy == "predicted_argmax"
                else targets
            )
            masked_logits = logits.double().clone()
            masked_logits.scatter_(1, mask_targets[:, None], float("-inf"))
            probabilities = torch.softmax(masked_logits, dim=1)
            class_sum.index_add_(0, targets, probabilities)
            class_count.index_add_(
                0, targets, torch.ones_like(targets, dtype=torch.float64)
            )

    if torch.any(class_count == 0):
        missing = torch.where(class_count == 0)[0].tolist()
        raise RuntimeError(f"No Stage 1 samples for classes: {missing}")
    class_average = class_sum / class_count[:, None]
    similarity = normalize_and_symmetrize_similarity(class_average).float().cpu()
    if not torch.isfinite(similarity).all():
        raise FloatingPointError("Similarity matrix contains NaN or Inf")
    save_cache(cache_path, similarity, metadata)
    torch.save(
        {
            "class_count": class_count.cpu(),
            "class_average": class_average.float().cpu(),
            "similarity": similarity,
            "mask_strategy": args.mask_strategy,
        },
        os.path.join(args.run_dir, "stage1_diagnostics.pt"),
    )
    print(f"Saved audited similarity cache: {cache_path}")
    return similarity
