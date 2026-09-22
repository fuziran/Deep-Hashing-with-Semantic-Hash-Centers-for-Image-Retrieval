"""Stage 3 auxiliary supervision with frozen, audited B1 artifacts."""
import hashlib
import math
import os

import numpy as np
import torch
from torch import nn

from utils.experiment import build_cache_metadata, stable_config_hash
from utils.reliable_relation import tensor_sha256


def mapping_from_labels(fine_labels, coarse_labels):
    fine = np.asarray(fine_labels)
    coarse = np.asarray(coarse_labels)
    if (fine.ndim != 1 or coarse.shape != fine.shape or not fine.size
            or fine.dtype.kind not in "iu" or coarse.dtype.kind not in "iu"
            or np.any((fine < 0) | (fine >= 100))
            or np.any((coarse < 0) | (coarse >= 20))):
        raise ValueError("Expected integer CIFAR-100 fine/coarse label pairs")
    mapping = np.full(100, -1, dtype=np.int64)
    for class_id in range(100):
        groups = np.unique(coarse[fine == class_id])
        if len(groups) != 1:
            raise ValueError(f"Missing or inconsistent coarse mapping for fine class {class_id}")
        mapping[class_id] = groups[0]
    if not np.all(np.bincount(mapping, minlength=20) == 5):
        raise ValueError("CIFAR-100 requires exactly five fine classes per coarse class")
    return mapping.tolist()


def load_official_mapping(root):
    # Read taxonomy from the official training file only; no query-image inference.
    from data.data_loader import _resolve_cifar100_root, _unpickle
    raw = _unpickle(os.path.join(_resolve_cifar100_root(root), "train"))
    return mapping_from_labels(raw["fine_labels"], raw["coarse_labels"])


def auxiliary_weight(max_weight, warmup_epochs, epoch):
    if not math.isfinite(max_weight) or max_weight < 0 or warmup_epochs < 0 or epoch < 0:
        raise ValueError("Invalid auxiliary weight schedule")
    return max_weight * min(1.0, (epoch + 1) / warmup_epochs) if warmup_epochs else max_weight


def make_auxiliary_head(net, weight, target):
    if weight == 0:
        return None
    # Preserve the baseline RNG stream (including subsequent loss initialization).
    with torch.random.fork_rng(devices=[]):
        return nn.Linear(net.hash_layer.in_features, 20 if target == "coarse" else 100)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_centers(args, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"value", "metadata"} <= payload.keys():
        raise RuntimeError("Expected an audited B1 center cache with value and metadata")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise RuntimeError("Invalid center metadata")
    unhashed = dict(metadata)
    stored_hash = unhashed.pop("config_hash", None)
    if stored_hash != stable_config_hash(unhashed):
        raise RuntimeError("Invalid center metadata hash")
    expected = build_cache_metadata(args, "centers")
    mismatch = {key: (value, metadata.get(key)) for key, value in expected.items()
                if key not in {"git_commit", "config_hash"} and value != metadata.get(key)}
    if mismatch:
        raise RuntimeError(f"Fixed B1 center cache is incompatible: {mismatch}")
    centers = payload["value"]
    if not isinstance(centers, torch.Tensor) or centers.shape not in (
            (args.code_length, args.num_classes), (args.num_classes, args.code_length)):
        raise RuntimeError("Invalid fixed center dimensions")
    if not torch.isfinite(centers).all() or not ((centers == -1) | (centers == 1)).all():
        raise RuntimeError("Fixed centers must be finite binary {-1,+1}")
    args.center_cache_config_hash = stored_hash
    args.center_cache_git_commit = metadata.get("git_commit")
    args.center_content_sha256 = tensor_sha256(centers)
    print(f"Loaded frozen B1 centers: {path}; sha256={args.center_content_sha256}")
    return centers.detach().clone()


def hierarchy_errors(query_binary, query_labels, database_binary, database_labels,
                     mapping, topks=(100, 1000)):
    """Diagnostic only: preserve exact Hamming ranking and stable tie handling."""
    q = np.asarray(query_binary, dtype=np.float32)
    d = np.asarray(database_binary, dtype=np.float32)
    qf, df = np.asarray(query_labels).argmax(1), np.asarray(database_labels).argmax(1)
    groups = np.asarray(mapping)
    totals = {k: np.zeros(3, dtype=np.int64) for k in topks}
    for code, fine in zip(q, qf):
        order = np.argsort(0.5 * (q.shape[1] - code @ d.T), kind="stable")
        for k in topks:
            selected = df[order[:k]]
            wrong = selected != fine
            same = groups[selected] == groups[fine]
            totals[k] += (np.count_nonzero(wrong & same),
                          np.count_nonzero(wrong & ~same), len(selected))
    return {f"top{k}": {
        "same_coarse_wrong_fine_count": int(v[0]),
        "cross_coarse_error_count": int(v[1]),
        "retrieved_count": int(v[2]),
        "same_coarse_wrong_fine_rate": float(v[0] / max(v[2], 1)),
        "cross_coarse_error_rate": float(v[1] / max(v[2], 1)),
    } for k, v in totals.items()}
