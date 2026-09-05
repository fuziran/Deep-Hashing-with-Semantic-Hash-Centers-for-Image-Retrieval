import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from network import ClassifyNet
from utils.experiment import (
    build_cache_metadata,
    load_cache,
    save_cache,
    state_dict_sha256,
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


def select_temperature(logits, targets, candidates):
    """Choose a scalar temperature by validation NLL; candidates must include 1."""
    candidates = [float(value) for value in candidates]
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("temperature candidates must be positive")
    if not any(abs(value - 1.0) < 1e-12 for value in candidates):
        raise ValueError("temperature candidates must include 1.0")
    logits = logits.double()
    losses = {
        value: float(F.cross_entropy(logits / value, targets).item())
        for value in candidates
    }
    best = min(candidates, key=lambda value: (losses[value], value))
    return best, losses


def prediction_reliability(probabilities, eps=1e-12):
    probabilities = probabilities.clamp_min(eps)
    entropy = -(probabilities * probabilities.log()).sum(dim=1)
    reliability = 1.0 - entropy / math.log(probabilities.shape[1])
    return reliability.clamp(0.0, 1.0)


def weighted_class_average(values, targets, weights, num_classes, eps=1e-12):
    class_sum = torch.zeros(
        num_classes, values.shape[1], dtype=values.dtype, device=values.device
    )
    weight_sum = torch.zeros(num_classes, dtype=values.dtype, device=values.device)
    class_sum.index_add_(0, targets, values * weights[:, None])
    weight_sum.index_add_(0, targets, weights)
    if torch.any(weight_sum <= eps):
        missing = torch.where(weight_sum <= eps)[0].tolist()
        raise RuntimeError(f"Zero reliable weight for classes: {missing}")
    return class_sum / weight_sum[:, None], weight_sum


def build_rsm_similarity(confusion, prototype_similarity=None, alpha=0.7):
    """Normalize each view before fusion so ablations and alpha are interpretable."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("prototype alpha must be in [0, 1]")
    confusion_similarity = normalize_and_symmetrize_similarity(confusion)
    if prototype_similarity is None:
        return confusion_similarity
    prototype_affinity = ((prototype_similarity + 1.0) / 2.0).clamp(0.0, 1.0)
    prototype_affinity = (prototype_affinity + prototype_affinity.T) / 2
    prototype_affinity.fill_diagonal_(0.0)
    normalized_prototypes = normalize_and_symmetrize_similarity(prototype_affinity)
    fused = alpha * confusion_similarity + (1.0 - alpha) * normalized_prototypes
    fused = (fused + fused.T) / 2
    fused.fill_diagonal_(1.0)
    return fused.clamp(-1.0, 1.0)


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


def load_explicit_classifier_cache(args, path):
    """Reuse an audited B0 classifier across B1-only code revisions."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(("value", "metadata")) - set(payload):
        raise RuntimeError(f"Invalid explicit classifier cache: {path}")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Invalid classifier cache metadata: {path}")
    stored_without_hash = dict(metadata)
    stored_hash = stored_without_hash.pop("config_hash", None)
    if stored_hash != stable_config_hash(stored_without_hash):
        raise RuntimeError(f"Classifier cache metadata hash is invalid: {path}")

    expected = build_cache_metadata(args, "classifier")
    provenance_fields = {"git_commit", "config_hash"}
    if getattr(args, "method", "B0") != "B0":
        provenance_fields.add("classifier_sha256")
    mismatches = {
        key: (expected[key], metadata.get(key))
        for key in expected.keys() - provenance_fields
        if expected[key] != metadata.get(key)
    }
    if mismatches:
        raise RuntimeError(
            f"Explicit classifier cache is incompatible: {path}\n"
            f"mismatches={mismatches}"
        )
    state = payload["value"]
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Invalid classifier state_dict: {path}")
    args.classifier_sha256 = state_dict_sha256(state)
    print(f"Loaded explicit audited classifier cache: {path}")
    print(
        "Classifier provenance: "
        f"git_commit={metadata.get('git_commit')} config_hash={stored_hash} "
        f"state_sha256={args.classifier_sha256}"
    )
    return state


def TrainClassificationNetwork(args, train_loader, val_loader):
    print("========== start classification network ==========")
    if args.classifier_cache:
        cached_state = load_explicit_classifier_cache(args, args.classifier_cache)
        net = ClassifyNet(args.num_classes).to(args.device)
        net.load_state_dict(cached_state)
        return net
    cache_path, metadata = _classifier_cache(args)
    if not args.force_recompute:
        cached_state = load_cache(cache_path, metadata)
        if cached_state is not None:
            net = ClassifyNet(args.num_classes).to(args.device)
            net.load_state_dict(cached_state)
            args.classifier_sha256 = state_dict_sha256(cached_state)
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
    args.classifier_sha256 = state_dict_sha256(best_state)
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
    if getattr(args, "method", "B0") != "B0":
        provenance_fields.add("classifier_sha256")
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

    cached_value = payload["value"]
    if isinstance(cached_value, dict) and "similarity" in cached_value:
        similarity = cached_value["similarity"]
        diagnostics = cached_value.get("diagnostics")
        if isinstance(diagnostics, dict) and "temperature" in diagnostics:
            args.rsm_selected_temperature = diagnostics["temperature"]
            if hasattr(args, "run_dir"):
                torch.save(
                    diagnostics,
                    os.path.join(args.run_dir, "stage1_diagnostics.pt"),
                )
    else:
        similarity = cached_value
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


def _collect_validation_logits(net, val_loader, device):
    all_logits, all_targets = [], []
    net.eval()
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            targets = labels.argmax(dim=1).to(device, non_blocking=True)
            logits, _ = net(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    return torch.cat(all_logits), torch.cat(all_targets)


def _generate_rsm(args, net, relation_loader, val_loader):
    if args.method in {"A2", "A3", "B1"}:
        validation_logits, validation_targets = _collect_validation_logits(
            net, val_loader, args.device
        )
        temperature, temperature_losses = select_temperature(
            validation_logits,
            validation_targets,
            args.rsm_temperature_grid,
        )
    else:
        temperature = 1.0
        temperature_losses = {1.0: float("nan")}
    args.rsm_selected_temperature = temperature

    probability_batches = []
    feature_batches = []
    target_batches = []
    index_batches = []
    net.eval()
    with torch.no_grad():
        for views, labels, indices in relation_loader:
            if views.ndim != 5:
                raise RuntimeError(
                    "RSM relation loader must return [batch, views, channels, height, width]"
                )
            batch_size, num_views = views.shape[:2]
            flat_views = views.flatten(0, 1).to(args.device, non_blocking=True)
            targets = labels.argmax(dim=1).to(args.device, non_blocking=True)
            logits, features = net.forward_features(flat_views)
            logits = logits.reshape(batch_size, num_views, -1).double()
            features = features.reshape(batch_size, num_views, -1).double()
            if args.mask_strategy == "predicted_argmax":
                mask_targets = logits.argmax(dim=2)
            else:
                mask_targets = targets[:, None].expand(-1, num_views)
            masked_logits = logits.clone()
            masked_logits.scatter_(2, mask_targets[:, :, None], float("-inf"))
            probabilities = torch.softmax(masked_logits / temperature, dim=2).mean(dim=1)
            normalized_features = F.normalize(features, dim=2)
            sample_features = F.normalize(normalized_features.mean(dim=1), dim=1)
            probability_batches.append(probabilities.cpu())
            feature_batches.append(sample_features.cpu())
            target_batches.append(targets.cpu())
            index_batches.append(indices.cpu())

    probabilities = torch.cat(probability_batches)
    features = torch.cat(feature_batches)
    targets = torch.cat(target_batches)
    sample_indices = torch.cat(index_batches)
    if args.method in {"A3", "B1"}:
        reliability = prediction_reliability(probabilities)
    else:
        reliability = torch.ones(len(targets), dtype=probabilities.dtype)

    confusion, class_weight = weighted_class_average(
        probabilities, targets, reliability, args.num_classes
    )
    prototype_similarity = None
    prototypes = None
    if args.method == "B1":
        prototypes, _ = weighted_class_average(
            features, targets, reliability, args.num_classes
        )
        prototypes = F.normalize(prototypes, dim=1)
        prototype_similarity = prototypes @ prototypes.T
    similarity = build_rsm_similarity(
        confusion,
        prototype_similarity,
        alpha=args.rsm_confusion_alpha,
    ).float()
    diagnostics = {
        "method": args.method,
        "temperature": temperature,
        "temperature_nll": temperature_losses,
        "sample_indices": sample_indices,
        "reliability": reliability.float(),
        "class_weight": class_weight.float(),
        "confusion": confusion.float(),
        "prototypes": None if prototypes is None else prototypes.float(),
        "prototype_similarity": (
            None if prototype_similarity is None else prototype_similarity.float()
        ),
        "similarity": similarity,
    }
    return similarity, diagnostics


def GenerateSimilarityMatrix(args, train_loader, relation_loader, val_loader):
    if args.similarity_cache:
        return load_explicit_similarity_cache(args, args.similarity_cache)
    if args.method != "B0":
        net = TrainClassificationNetwork(args, train_loader, val_loader)
        cache_path, metadata = _similarity_cache(args)
        args.similarity_hash = metadata["config_hash"]
        if not args.force_recompute:
            cached = load_cache(cache_path, metadata)
            if cached is not None:
                if not isinstance(cached, dict) or "similarity" not in cached:
                    raise RuntimeError(f"Invalid audited RSM cache payload: {cache_path}")
                diagnostics = cached.get("diagnostics")
                if isinstance(diagnostics, dict):
                    args.rsm_selected_temperature = diagnostics.get("temperature")
                    torch.save(
                        diagnostics,
                        os.path.join(args.run_dir, "stage1_diagnostics.pt"),
                    )
                print(f"Loaded audited RSM similarity cache: {cache_path}")
                return cached["similarity"]
        print(f"========== start {args.method} robust similarity matrix ==========")
        similarity, diagnostics = _generate_rsm(
            args, net, relation_loader, val_loader
        )
        if not torch.isfinite(similarity).all():
            raise FloatingPointError("RSM similarity matrix contains NaN or Inf")
        if similarity.min() < -1.000001 or similarity.max() > 1.000001:
            raise FloatingPointError("RSM similarity matrix is outside [-1, 1]")
        save_cache(
            cache_path,
            {"similarity": similarity.cpu(), "diagnostics": diagnostics},
            metadata,
        )
        torch.save(diagnostics, os.path.join(args.run_dir, "stage1_diagnostics.pt"))
        print(f"Saved audited RSM similarity cache: {cache_path}")
        return similarity.cpu()

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
