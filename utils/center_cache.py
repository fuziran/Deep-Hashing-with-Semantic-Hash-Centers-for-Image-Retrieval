"""Explicit audited center reuse; never regenerate centers for C1."""
import torch

from utils.experiment import build_cache_metadata, stable_config_hash, state_dict_sha256
from GenerateSemanticHashCenters import get_margin, pairwise_hamming_distances


def load_explicit_centers(args, path):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise RuntimeError("Invalid audited center cache")
    metadata = dict(payload["metadata"])
    stored_hash = metadata.pop("config_hash", None)
    if stable_config_hash(metadata) != stored_hash:
        raise RuntimeError("Center metadata hash is invalid")
    expected = build_cache_metadata(args, "centers")
    mismatches = {k: (v, metadata.get(k)) for k, v in expected.items()
                  if k not in {"git_commit", "config_hash"} and metadata.get(k) != v}
    if mismatches:
        raise RuntimeError(f"Center protocol mismatch: {mismatches}")
    centers = payload.get("value")
    if not isinstance(centers, torch.Tensor) or centers.shape != (args.code_length, args.num_classes):
        raise RuntimeError("Center shape must be [bits, classes]")
    if not torch.all((centers == -1) | (centers == 1)):
        raise RuntimeError("Centers must contain only -1 and +1")
    minimum, _ = get_margin(args.code_length, args.num_classes)
    if pairwise_hamming_distances(centers).min().item() < minimum:
        raise RuntimeError("Centers violate the minimum Hamming distance")
    args.centers_sha256 = state_dict_sha256({"centers": centers})
    args.centers_config_hash = stored_hash
    print(f"Loaded audited fixed B0 centers: {path}; SHA256={args.centers_sha256}")
    return centers
