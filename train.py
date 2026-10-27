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

from utils.tools import CalcTopMapPerQuery, CalcTopMapWithPR, compute_result
from utils.competition import CenterCompetition, competition_weight
from utils.bit_supervision import BitSupervision, risk_schedule
from utils.semantic_mixup import SemanticMixup, mix_weight, backward_mix
from utils.semantic_rank import SemanticRank, rank_weight, sampled_gradient_diagnostics
from utils.rank_checkpoint import atomic_save, save_checkpoint, restore_checkpoint

torch.multiprocessing.set_sharing_strategy("file_system")


class CSQLoss(torch.nn.Module):
    def __init__(self, args, bit, hash_center, similarity=None):
        super().__init__()
        self.register_buffer("hash_targets", hash_center.detach().to(args.device))
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(args.device)
        self.criterion = torch.nn.BCELoss().to(args.device)
        self.competition = None
        self.bit_supervision = None
        self.epoch = 1
        self.components = {}
        if getattr(args, "variant", "b0") == "b2":
            if similarity is None:
                raise ValueError("B2 requires the frozen B1 relation matrix")
            self.competition = CenterCompetition(
                self.hash_targets, similarity, args.comp_temperature, args.comp_m0, args.comp_m1
            )

        if getattr(args, "variant", "b0") == "b2r":
            if similarity is None:
                raise ValueError("B2R requires the frozen B1 relation matrix")
            self.bit_supervision = BitSupervision(
                self.hash_targets, similarity, args.i2r_stats_start, args.i2r_warmup
            )

    def forward(self, output, labels, indices, args):
        del indices
        continuous_code = output.tanh()
        target_center = self.label2center(labels)
        center_loss = self.criterion(
            0.5 * (continuous_code + 1), 0.5 * (target_center + 1)
        )
        uniform_center_loss = center_loss
        rho = 0.0
        if self.bit_supervision is not None:
            targets = labels.argmax(dim=1)
            self.bit_supervision.observe(continuous_code, targets, self.epoch)
            rho = risk_schedule(self.epoch, args.i2r_rho, args.i2r_stats_start,
                                args.i2r_warmup, args.i2r_ramp_end)
            # Exact B1 reduction while disabled, including rho=0 runs.
            if rho > 0:
                per_bit = torch.nn.functional.binary_cross_entropy(
                    0.5 * (continuous_code + 1), 0.5 * (target_center + 1),
                    reduction="none",
                )
                weights = self.bit_supervision.weights(targets, rho, per_bit)
                center_loss = (weights * per_bit).mean()
        quantization_loss = (continuous_code.abs() - 1).pow(2).mean()
        loss = center_loss + args.lambd * quantization_loss
        comp_loss = continuous_code.new_zeros(())
        weight = 0.0
        if self.competition is not None:
            comp_loss = self.competition(continuous_code, labels.argmax(dim=1))
            weight = competition_weight(
                self.epoch, args.comp_weight, args.comp_warmup, args.comp_ramp_end
            )
            loss = loss + weight * comp_loss
        self.components = {
            "loss_center": center_loss.detach().item(),
            "loss_quant": quantization_loss.detach().item(),
            "loss_comp": comp_loss.detach().item(),
            "comp_weight": weight,
        }
        if self.bit_supervision is not None:
            self.components.update({"loss_center_uniform": uniform_center_loss.detach().item(),
                                    "i2r_rho": rho})
        return loss

    def finish_epoch(self, args):
        if self.bit_supervision is None or self.epoch != args.i2r_warmup:
            return
        self.bit_supervision.freeze()
        diagnostics = self.bit_supervision.diagnostics(args.i2r_rho)
        os.makedirs(args.run_dir, exist_ok=True)
        path = os.path.join(args.run_dir, "i2r_statistics.pt")
        torch.save(diagnostics, path)
        with open(os.path.join(args.run_dir, "i2r_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump({key: value for key, value in diagnostics.items() if key != "tensors"},
                      f, indent=2)
        args.i2r_statistics_path = path
        args.i2r_statistics_sha256 = diagnostics["content_sha256"]
        from utils.experiment import write_run_manifest
        write_run_manifest(args)
        print(f"{args.info} I2-R statistics frozen: " + json.dumps(diagnostics["summary"]))

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


def train_val(
    args,
    hash_center,
    train_loader,
    val_loader,
    query_loader,
    database_loader,
    num_database,
    similarity=None,
    nfd=None,
    npa=None,
    fcd=None,
):
    print(f"========== start {args.info} network training ==========")
    bit = args.code_length
    if (getattr(args, "variant", "b0") in ("b2nfd", "b3npa", "b3fcd")) != (nfd is not None):
        raise ValueError("NFD variant requires exactly one prepared frozen distiller")
    if npa is not None and getattr(args, "variant", "b0") != "b3npa":
        raise ValueError("NPA requires the b3npa variant")
    if (getattr(args, "variant", "b0") == "b3npa" and args.npa_weight > 0
            and args.npa_rho > 0 and npa is None):
        raise ValueError("Enabled NPA requires prepared training-only statistics")
    net = args.net(bit).to(args.device)
    if fcd is not None and getattr(args,'variant','b0') != 'b3fcd':
        raise ValueError('FCD requires b3fcd')
    if getattr(args,'variant','b0') == 'b3fcd' and args.fcd_weight > 0 and fcd is None:
        raise ValueError('Enabled FCD requires a validated teacher cache')
    parameters = list(net.parameters())
    if fcd is not None:
        parameters += list(fcd.head.parameters())
    optimizer = optim.RMSprop(parameters, lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch, eta_min=1e-7
    )

    hash_center = hash_center.float()
    if hash_center.shape != (args.num_classes, bit):
        hash_center = hash_center.T
    criterion = CSQLoss(args, bit, hash_center, similarity)
    mixer = None
    if getattr(args, "variant", "b0") == "b2m":
        if similarity is None:
            raise ValueError("B2M requires the frozen B1 relation matrix")
        mixer = SemanticMixup(hash_center, similarity, args.mix_seed,
                              args.mix_mode, args.mix_budget)

    ranker = None
    if getattr(args, "variant", "b0") == "b2sbr":
        if similarity is None:
            raise ValueError("SBR requires the frozen B1 relation matrix")
        ranker = SemanticRank(hash_center, similarity, args.rank_capacity,
                              args.rank_max_age, args.rank_mode, args.rank_chunk_size)
        os.makedirs(args.run_dir, exist_ok=True)

    best_validation_map = float("-inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    history, losses, learning_rates = [], [], []

    start_epoch, global_step = 0, 0
    if ranker is not None and getattr(args, "resume_sbr", None):
        resumed = restore_checkpoint(args.resume_sbr, args, net, optimizer, scheduler, ranker, train_loader)
        start_epoch, global_step = resumed["epoch"], resumed["global_step"]
        best_state, best_epoch = resumed["best_state"], resumed["best_epoch"]
        best_validation_map, history = resumed["best_map"], resumed["history"]
        losses = [row["loss"] for row in history]
        learning_rates = [row["lr"] for row in history]
        atomic_save(best_state, os.path.join(args.run_dir, "best_model_state.pt"))

    for epoch in range(start_epoch, args.epoch):
        started = time.time()
        learning_rates.append(optimizer.param_groups[0]["lr"])
        net.train()
        total_loss = 0.0
        component_totals = {}
        criterion.epoch = epoch + 1
        if mixer is not None:
            mixer.reset_diagnostics()
        if ranker is not None:
            ranker.reset_diagnostics()
        gradient_sample = {}
        for batch_index, (images, labels, indices) in enumerate(train_loader):
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            nfd_strength = 0.0
            if nfd is not None:
                from utils.neighbor_feature_distill import nfd_weight
                nfd_strength = nfd_weight(epoch + 1, args.nfd_weight,
                                          args.nfd_warmup, args.nfd_ramp_end)
            npa_strength = 0.0
            if npa is not None:
                from utils.neighborhood_protected_augmentation import npa_weight
                npa_strength = npa_weight(epoch + 1, args.npa_weight,
                                          args.npa_warmup, args.npa_ramp_end)
            if nfd_strength > 0 or npa_strength > 0:
                output, features = net.forward_with_features(images)
            else:
                output = net(images)
            loss = criterion(output, labels, indices, args)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite hash loss at epoch {epoch + 1}")
            auxiliary_loss = 0.0
            rank_strength = 0.0
            if ranker is not None:
                rank_strength = rank_weight(epoch + 1, args.rank_weight,
                                            args.rank_warmup, args.rank_ramp_end)
                if rank_strength > 0:
                    rank_loss = ranker.loss(output.tanh(), labels.argmax(dim=1), indices, global_step)
                    if not torch.isfinite(rank_loss):
                        raise FloatingPointError("Non-finite binary ranking loss")
                    weighted_rank = rank_strength * rank_loss
                    if batch_index == 0:
                        gradient_sample = sampled_gradient_diagnostics(loss, weighted_rank, net.parameters())
                    (loss + weighted_rank).backward()
                    auxiliary_loss = weighted_rank.detach().item()
                    criterion.components["loss_rank"] = rank_loss.detach().item()
                else:
                    loss.backward()
                    criterion.components["loss_rank"] = 0.0
                criterion.components["rank_weight"] = rank_strength
            elif nfd is not None:
                combined = loss
                if nfd_strength > 0:
                    nfd_loss = nfd(features, images, labels.argmax(dim=1))
                    if not torch.isfinite(nfd_loss):
                        raise FloatingPointError("Non-finite NFD loss")
                    weighted_nfd = nfd_strength * nfd_loss
                    if batch_index == 0:
                        sample = sampled_gradient_diagnostics(loss, weighted_nfd, net.parameters())
                        gradient_sample = {key.replace("rank", "nfd"): value for key, value in sample.items()}
                    combined = combined + weighted_nfd
                    auxiliary_loss = weighted_nfd.detach().item()
                    criterion.components.update(nfd.components)
                else:
                    criterion.components["loss_nfd"] = 0.0
                criterion.components["nfd_weight"] = nfd_strength
                if npa_strength > 0:
                    npa_loss = npa(output, features, labels.argmax(dim=1),
                                   net.hash_layer.weight, criterion.hash_targets)
                    weighted_npa = npa_strength * npa_loss
                    if batch_index == 0:
                        sample = sampled_gradient_diagnostics(combined, weighted_npa, net.parameters())
                        gradient_sample["sampled_npa_reference_grad_norm"] = sample["sampled_clean_grad_norm"]
                        gradient_sample.update({key.replace("rank", "npa"): value for key, value in sample.items()
                                                if key != "sampled_clean_grad_norm"})
                    combined = combined + weighted_npa
                    auxiliary_loss += weighted_npa.detach().item()
                    criterion.components.update(npa.components)
                if getattr(args, "variant", "b0") == "b3npa":
                    criterion.components.setdefault("loss_npa", 0.0)
                    criterion.components["npa_weight"] = npa_strength
                if fcd is not None:
                    from utils.foundation_distill import ramp
                    fcd_strength = ramp(epoch+1,args.fcd_weight,args.fcd_warmup,args.fcd_ramp_end)
                    if fcd_strength > 0:
                        extra = fcd(output.tanh(),labels.argmax(1),indices)
                        weighted_fcd = fcd_strength*extra
                        if batch_index == 0:
                            sample = sampled_gradient_diagnostics(combined,weighted_fcd,net.parameters())
                            gradient_sample.update({'fcd_'+key: value for key,value in sample.items()})
                        combined = combined+weighted_fcd
                        auxiliary_loss += weighted_fcd.detach().item()
                        criterion.components.update(fcd.components)
                    else:
                        criterion.components['loss_fcd'] = 0.
                    criterion.components['fcd_weight'] = fcd_strength
                combined.backward()
            else:
                loss.backward()
            if mixer is not None:
                weight = mix_weight(epoch + 1, args.mix_weight,
                                    args.mix_warmup, args.mix_ramp_end)
                mixed_components = backward_mix(net, images, labels.argmax(dim=1),
                                                mixer, weight, batch_index == 0)
                gradient_sample = {key: value for key, value in mixed_components.items()
                                   if key.startswith("sampled_")} or gradient_sample
                criterion.components.update({key: value for key, value in mixed_components.items()
                                             if not key.startswith("sampled_")})
                auxiliary_loss = weight * mixed_components["loss_mix"]
            optimizer.step()
            if ranker is not None:
                if rank_strength > 0:
                    ranker.enqueue(output.detach().tanh(), labels.argmax(dim=1), indices, global_step)
                global_step += 1
            total_loss += loss.item() + auxiliary_loss
            for key, value in criterion.components.items():
                component_totals[key] = component_totals.get(key, 0.0) + value
        criterion.finish_epoch(args)
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
        record.update({key: value / max(len(train_loader), 1)
                       for key, value in component_totals.items()})
        if mixer is not None:
            record.update(gradient_sample)
            diagnostics_dir = os.path.join(args.run_dir, "mix_diagnostics")
            os.makedirs(diagnostics_dir, exist_ok=True)
            diagnostics_path = os.path.join(diagnostics_dir, f"epoch_{epoch + 1:03d}.json")
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(mixer.diagnostics(), f, indent=2)
            record["mix_diagnostics"] = os.path.relpath(diagnostics_path, args.run_dir)
            # A sampler state is not a full optimizer/model resume checkpoint.
            torch.save({"epoch": epoch + 1, "sampler": mixer.state_dict()},
                       os.path.join(args.run_dir, "mix_sampler_state.pt"))
        if ranker is not None:
            record.update(gradient_sample)
            record["rank_diagnostics"] = ranker.diagnostics()
            print(f"{args.info} epoch={epoch + 1} rank_diagnostics="
                  + json.dumps(record["rank_diagnostics"]))
            if gradient_sample:
                print(f"{args.info} epoch={epoch + 1} rank_gradient_sample="
                      + json.dumps(gradient_sample))
        if nfd is not None:
            record.update(gradient_sample)
            if gradient_sample:
                print(f"{args.info} epoch={epoch + 1} nfd_gradient_sample=" + json.dumps(gradient_sample))
        if (criterion.competition is not None or criterion.bit_supervision is not None
                or mixer is not None or ranker is not None or nfd is not None):
            print(f"{args.info} epoch={epoch + 1} components="
                  + json.dumps({key: record[key] for key in component_totals}))
        if should_validate:
            validation_metrics, _, _, _, _, _ = _evaluate(
                val_loader, database_loader, net, args, num_database
            )
            record["validation"] = validation_metrics
            if validation_metrics["mAP@ALL"] > best_validation_map:
                best_validation_map = validation_metrics["mAP@ALL"]
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone() for key, value in net.state_dict().items()
                }
                if ranker is not None or nfd is not None:
                    atomic_save(best_state, os.path.join(args.run_dir, "best_model_state.pt"))
                if fcd is not None:
                    atomic_save({key:value.detach().cpu().clone() for key,value in fcd.head.state_dict().items()},
                                os.path.join(args.run_dir,'fcd_selected_auxiliary_head.pt'))
            print(
                f"{args.info} epoch={epoch + 1} loss={epoch_loss:.5f} "
                f"val_ALL={validation_metrics['mAP@ALL']:.5f} "
                f"val_100={validation_metrics['mAP@100']:.5f} "
                f"val_1000={validation_metrics['mAP@1000']:.5f}"
            )
        else:
            print(f"{args.info} epoch={epoch + 1} loss={epoch_loss:.5f}")
        history.append(record)
        if ranker is not None:
            record["seconds_before_checkpoint"] = time.time() - started
            save_checkpoint(os.path.join(args.run_dir, "last_checkpoint.pt"), args, net,
                            optimizer, scheduler, ranker, train_loader, epoch + 1, global_step,
                            best_state, best_epoch, best_validation_map, history)
        if mixer is not None or ranker is not None or nfd is not None:
            with open(os.path.join(args.run_dir, "training_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    # Save selection before any optional final-query evaluation.
    os.makedirs(args.run_dir, exist_ok=True)
    torch.save(best_state, os.path.join(args.run_dir, "best_model_state.pt"))
    with open(os.path.join(args.run_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    from utils.experiment import selection_receipt
    receipt = selection_receipt(args.run_dir)
    if getattr(args, "skip_query", False):
        summary = {
            **receipt,
            "selected_epoch": best_epoch,
            "selection_metric": "validation mAP@ALL",
            "best_validation_mAP@ALL": best_validation_map,
            "query_evaluated": False,
            "query_metrics": None,
        }
        with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _save_curves(args.run_dir, losses, learning_rates)
        print(f"Selected epoch {best_epoch}; final query evaluation skipped")
        return summary
    net.load_state_dict(best_state)
    net.to(args.device)
    final_metrics, pr_data, query_binary, query_label, database_binary, database_label = (
        _evaluate(query_loader, database_loader, net, args, num_database)
    )
    per_query_ap = CalcTopMapPerQuery(
        query_binary.numpy(),
        query_label.numpy(),
        database_binary.numpy(),
        database_label.numpy(),
        args.topK,
    )
    summary = {
        **receipt,
        "selected_epoch": best_epoch,
        "selection_metric": "validation mAP@ALL",
        "best_validation_mAP@ALL": best_validation_map,
        "query_metrics": final_metrics,
        "query_evaluated": True,
    }

    os.makedirs(args.run_dir, exist_ok=True)
    torch.save(
        {
            "query_binary": query_binary,
            "query_label": query_label,
            "database_binary": database_binary,
            "database_label": database_label,
            "pr_data": pr_data,
            "per_query_ap": per_query_ap,
        },
        os.path.join(args.run_dir, "retrieval_outputs.pt"),
    )
    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_curves(args.run_dir, losses, learning_rates)

    print(
        f"Selected epoch {best_epoch}; query mAP@ALL={final_metrics['mAP@ALL']:.5f}, "
        f"mAP@100={final_metrics['mAP@100']:.5f}, "
        f"mAP@1000={final_metrics['mAP@1000']:.5f}"
    )
    print(f"========== finished {args.info} network training ==========")
    return summary
