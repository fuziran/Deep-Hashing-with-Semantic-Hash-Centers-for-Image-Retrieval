"""I1: frozen two-view relation estimation using training images only."""
import hashlib
import math
import os

import torch
import torch.nn.functional as F

MODES = ("full", "two_view", "weighted", "fixed_fusion", "feature_only")
VERSION = "i1-relation-v1"


def tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(tensor_sha256(state[key]).encode())
    return digest.hexdigest()


def relation_config(args):
    config = {
        "version": VERSION,
        "mode": args.relation_mode,
        "kappa": args.relation_kappa,
        "temperature": args.relation_temperature,
        "weight_floor": args.relation_weight_floor,
        "views": ["resize_center_crop", "resize_center_crop_horizontal_flip"],
        "resize_size": args.resize_size,
        "crop_size": args.crop_size,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "feature_source": "ClassifyNet.feature_layers.pooled_flatten",
    }
    if getattr(args, "aux_manifest", None):
        config["auxiliary"] = {
            "version": "foreground-aux-v1", "alpha": args.aux_alpha,
            "content_sha256": args.aux_content_sha256, "count": args.aux_count,
            "branch": "probability_only", "view": "resize_center_crop_single",
            "quality": "manifest_approved_binary",
        }
    return config


def validate_similarity(matrix, classes):
    if not isinstance(matrix, torch.Tensor) or matrix.shape != (classes, classes):
        raise ValueError("Invalid relation matrix shape")
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("Relation matrix contains NaN or Inf")
    if (matrix.abs() > 1.000001).any():
        raise ValueError("Relation values outside [-1, 1]")
    if not torch.allclose(matrix, matrix.T, atol=1e-7, rtol=1e-6):
        raise ValueError("Relation matrix is not symmetric")
    if not torch.allclose(matrix.diag(), torch.ones_like(matrix.diag()), atol=1e-7):
        raise ValueError("Relation matrix diagonal must be one")


def view_statistics(logits1, logits2, targets, temperature=1.0,
                    weight_floor=0.05, mask_strategy="predicted_argmax"):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(weight_floor) or not 0 < weight_floor <= 1:
        raise ValueError("weight_floor must be in (0, 1]")
    if mask_strategy not in ("predicted_argmax", "ground_truth"):
        raise ValueError("Unknown mask strategy")
    if logits1.shape != logits2.shape or logits1.ndim != 2 or logits1.shape[1] < 2:
        raise ValueError("Expected matching [batch, classes>=2] logits")
    if not torch.isfinite(logits1).all() or not torch.isfinite(logits2).all():
        raise FloatingPointError("Non-finite classifier logits")
    logits1, logits2 = logits1.double(), logits2.double()
    p1, p2 = [(z / temperature).softmax(1) for z in (logits1, logits2)]
    midpoint = (p1 + p2) / 2
    log_mid = midpoint.clamp_min(1e-300).log()
    js = sum((p * (p.clamp_min(1e-300).log() - log_mid)).sum(1)
             for p in (p1, p2)) / 2
    consistency = (1 - js / math.log(2)).clamp(0, 1)
    confidence = (p1.gather(1, targets[:, None]).squeeze(1)
                  + p2.gather(1, targets[:, None]).squeeze(1)) / 2
    weight = (consistency * confidence).clamp_min(weight_floor)

    def masked(z, temp):
        mask = z.argmax(1) if mask_strategy == "predicted_argmax" else targets
        z = z.clone()
        z.scatter_(1, mask[:, None], float("-inf"))
        return (z / temp).softmax(1)

    return {
        "a1": masked(logits1, temperature),
        "a2": masked(logits2, temperature),
        # S0 is exactly the single original view with baseline T=1.
        "baseline": masked(logits1, 1.0),
        "weight": weight, "js": js, "consistency": consistency,
        "confidence": confidence,
        "agreement": (p1.argmax(1) == p2.argmax(1)).double(),
    }


def fuse_relations(probability, feature, effective_count, kappa):
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if not torch.isfinite(effective_count).all() or (effective_count <= 0).any():
        raise ValueError("Every class needs positive effective evidence")
    reliability = effective_count / (effective_count + kappa)
    gate = torch.sqrt(reliability[:, None] * reliability[None, :])
    # Do not normalize rows again: that would change the reliability gate.
    fused = (gate * probability + (1 - gate) * feature).clamp(-1, 1)
    fused.fill_diagonal_(1)
    validate_similarity(fused, len(effective_count))
    return fused, gate


@torch.no_grad()
def estimate_relations(net, loader, args):
    from GenerateSimilarityMatrix import normalize_and_symmetrize_similarity

    classes = args.num_classes
    if classes < 2 or args.relation_mode not in MODES:
        raise ValueError("I1 requires >=2 classes and a known relation mode")
    # CPU float64 accumulation avoids CUDA index_add atomic reduction order.
    sums = {name: torch.zeros(classes, classes, dtype=torch.float64)
            for name in ("baseline", "unweighted", "weighted", "original_weighted")}
    counts = torch.zeros(classes, dtype=torch.float64)
    stats = {name: torch.zeros_like(counts)
             for name in ("weight", "js", "consistency", "confidence", "agreement", "floor")}
    feature_sum = None
    auxiliary = getattr(getattr(loader, "dataset", None), "auxiliary", None)
    if getattr(args, "aux_manifest", None) and auxiliary is None:
        raise ValueError("Declared auxiliary views were not loaded")
    alpha = getattr(args, "aux_alpha", 0.0) if auxiliary else 0.0
    aux_rows, seen = [], set()
    net.eval()
    for images, labels, indices in loader:
        if (labels.ndim != 2 or labels.shape[1] != classes
                or not torch.isfinite(labels).all()
                or not ((labels == 0) | (labels == 1)).all()
                or not (labels.sum(1) == 1).all()):
            raise ValueError("I1 requires finite one-hot single-label targets")
        targets = labels.argmax(1).cpu()
        images = images.to(args.device, non_blocking=True)
        z1, f1 = net.forward_with_features(images)
        z2, f2 = net.forward_with_features(images.flip(-1))
        z1, z2, f1, f2 = [t.detach().cpu().double() for t in (z1, z2, f1, f2)]
        if z1.shape != (len(targets), classes):
            raise ValueError("Classifier output does not match class count")
        if f1.ndim != 2 or f1.shape != f2.shape or f1.shape[0] != len(targets):
            raise ValueError("Expected matching pooled visual features")
        if not torch.isfinite(f1).all() or not torch.isfinite(f2).all():
            raise FloatingPointError("Non-finite visual features")
        if (f1.norm(dim=1) <= 1e-12).any() or (f2.norm(dim=1) <= 1e-12).any():
            raise ValueError("Zero visual feature norm")
        batch = view_statistics(z1, z2, targets, args.relation_temperature,
                                args.relation_weight_floor, args.mask_strategy)
        averaged = (batch["a1"] + batch["a2"]) / 2
        sums["baseline"].index_add_(0, targets, batch["baseline"])
        sums["unweighted"].index_add_(0, targets, averaged)
        sums["original_weighted"].index_add_(0, targets, averaged * batch["weight"][:, None])
        if auxiliary:
            positions = [i for i, idx in enumerate(indices.tolist()) if idx in auxiliary["tensors"]]
            if positions:
                ids = [int(indices[i]) for i in positions]
                if seen.intersection(ids):
                    raise ValueError("Auxiliary training index repeated by relation loader")
                seen.update(ids)
                aux_images = torch.stack([auxiliary["tensors"][idx] for idx in ids]).to(args.device)
                aux_logits, _ = net.forward_with_features(aux_images)
                aux_logits = aux_logits.detach().cpu().double()
                aux_stats = view_statistics(aux_logits, aux_logits, targets[positions],
                                            args.relation_temperature, args.relation_weight_floor,
                                            args.mask_strategy)
                # alpha=0 deliberately skips arithmetic to preserve B1 bit-for-bit.
                if alpha != 0:
                    averaged = averaged.clone()
                    averaged[positions] = (1 - alpha) * averaged[positions] + alpha * aux_stats["a1"]
                original_probs = (z1[positions] / args.relation_temperature).softmax(1)
                aux_probs = (aux_logits / args.relation_temperature).softmax(1)
                for j, (pos, idx) in enumerate(zip(positions, ids)):
                    target = int(targets[pos])
                    aux_rows.append({"index": idx, "class_id": target,
                        "original_prediction": int(original_probs[j].argmax()),
                        "auxiliary_prediction": int(aux_probs[j].argmax()),
                        "original_true_class_confidence": float(original_probs[j, target]),
                        "auxiliary_true_class_confidence": float(aux_probs[j, target]),
                        "original_two_view_weight": float(batch["weight"][pos])})
        sums["weighted"].index_add_(0, targets, averaged * batch["weight"][:, None])
        features = (F.normalize(f1, dim=1, eps=1e-12)
                    + F.normalize(f2, dim=1, eps=1e-12)) / 2
        if feature_sum is None:
            feature_sum = torch.zeros(classes, features.shape[1], dtype=torch.float64)
        feature_sum.index_add_(0, targets, features)
        counts.index_add_(0, targets, torch.ones_like(targets, dtype=torch.float64))
        for name in stats:
            values = ((batch["weight"] <= args.relation_weight_floor).double()
                      if name == "floor" else batch[name])
            stats[name].index_add_(0, targets, values)

    if (counts == 0).any():
        raise ValueError(f"No Stage 1 samples for classes: {torch.where(counts == 0)[0].tolist()}")
    if auxiliary and seen != set(auxiliary["tensors"]):
        raise ValueError("Relation loader did not visit every declared auxiliary index")
    prototypes = feature_sum / counts[:, None]
    if (prototypes.norm(dim=1) <= 1e-12).any():
        raise ValueError("Zero class prototype norm")
    prototypes = F.normalize(prototypes, dim=1, eps=1e-12)
    norm = normalize_and_symmetrize_similarity
    s0 = norm(sums["baseline"] / counts[:, None])
    s_unweighted = norm(sums["unweighted"] / counts[:, None])
    s_prob = norm(sums["weighted"] / stats["weight"][:, None].clamp_min(1e-12))
    s_feat = norm(prototypes @ prototypes.T)
    s1, gate = fuse_relations(s_prob, s_feat, stats["weight"], args.relation_kappa)
    selected = {
        "full": s1, "two_view": s_unweighted, "weighted": s_prob,
        "fixed_fusion": 0.5 * (s_prob + s_feat), "feature_only": s_feat,
    }[args.relation_mode].float()
    for matrix in (s0, s_unweighted, s_prob, s_feat, s1, selected):
        validate_similarity(matrix, classes)
    artifact = {
        "similarity": selected, "S0": s0, "S_prob": s_prob,
        "S_feat": s_feat, "S1": s1, "S_two_view": s_unweighted,
        "class_count": counts, "e_c": stats["weight"], "gate": gate,
        "prototypes": prototypes,
        "view_statistics": {k: v / counts for k, v in stats.items()},
        "relation_config": relation_config(args),
    }
    if auxiliary:
        original_prob = norm(sums["original_weighted"] / stats["weight"][:, None].clamp_min(1e-12))
        original_s1, _ = fuse_relations(original_prob, s_feat, stats["weight"], args.relation_kappa)
        coverage = torch.bincount(torch.tensor([r["class_id"] for r in aux_rows]), minlength=classes)
        def differences(left, right):
            delta = (left - right).abs()
            return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                    "per_class_mean_abs": delta.mean(1).tolist()}
        artifact.update({"S_prob_original": original_prob, "S1_original": original_s1,
            "auxiliary_diagnostics": {"alpha": alpha, "count": len(aux_rows),
                "content_sha256": auxiliary["sha256"],
                "class_aux_count": coverage.tolist(),
                "class_train_count": counts.tolist(),
                "class_coverage": (coverage / counts).tolist(),
                "probability_relation_change": differences(s_prob, original_prob),
                "fused_relation_change": differences(s1, original_s1),
                "samples": sorted(aux_rows, key=lambda r: r["index"])}})
    return artifact


def generate_i1(args, train_loader, relation_loader, val_loader):
    from GenerateSimilarityMatrix import TrainClassificationNetwork
    from utils.experiment import build_cache_metadata, load_cache, save_cache

    auxiliary = getattr(getattr(relation_loader, "dataset", None), "auxiliary", None)
    if auxiliary:
        from utils.foreground_aux import save_previews
        save_previews(auxiliary, args.run_dir)
    net = TrainClassificationNetwork(args, train_loader, val_loader)
    args.classifier_sha256 = state_sha256(net.state_dict())
    metadata = build_cache_metadata(args, "similarity")
    path = os.path.join(args.output_dir, "SimilarityMatrix",
                        f"{args.dataset}_{metadata['config_hash']}.pt")
    artifact = None if args.force_recompute else load_cache(path, metadata)
    if artifact is None:
        artifact = estimate_relations(net, relation_loader, args)
        artifact["similarity_sha256"] = tensor_sha256(artifact["similarity"])
        artifact["classifier_sha256"] = args.classifier_sha256
        save_cache(path, artifact, metadata)
    return restore_artifact(args, artifact, metadata)


def restore_artifact(args, artifact, metadata):
    if not isinstance(artifact, dict) or "similarity" not in artifact:
        raise ValueError("Expected I1 cache with diagnostics and content digest")
    similarity = artifact["similarity"]
    validate_similarity(similarity, args.num_classes)
    digest = tensor_sha256(similarity)
    if artifact.get("similarity_sha256") != digest:
        raise RuntimeError("I1 similarity content SHA256 mismatch")
    if artifact.get("classifier_sha256") != metadata.get("classifier_sha256"):
        raise RuntimeError("I1 classifier provenance mismatch")
    if artifact.get("relation_config") != relation_config(args):
        raise RuntimeError("I1 relation configuration mismatch")
    args.similarity_hash = metadata["config_hash"]
    args.similarity_content_sha256 = digest
    args.classifier_sha256 = metadata["classifier_sha256"]
    torch.save(artifact, os.path.join(args.run_dir, "stage1_diagnostics.pt"))
    from utils.foreground_aux import export_diagnostics
    export_diagnostics(artifact, args.run_dir)
    print(f"I1 relation mode={args.relation_mode}, content sha256={digest}")
    return similarity