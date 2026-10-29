import hashlib
import json
import os
import random
import subprocess

import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)


def stable_config_hash(config, length=12):
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_receipt(run_dir):
    """Bind final selection to its model, manifest and complete training history."""
    files = {
        "checkpoint_sha256": "best_model_state.pt",
        "training_config_sha256": "config.json",
        "training_history_sha256": "training_history.json",
    }
    return {"selection_receipt_version": 1, **{
        key: file_sha256(os.path.join(run_dir, name))
        if os.path.isfile(os.path.join(run_dir, name)) else None
        for key, name in files.items()
    }}


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_sha256():
    """Identify local code even before the user commits the new branch."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for relative in (
        "run.py", "network.py", "GenerateSimilarityMatrix.py",
        "GenerateSemanticHashCenters.py", "train.py", "data/data_loader.py",
        "utils/experiment.py", "utils/tools.py", "utils/reliable_relation.py",
        "utils/competition.py", "utils/frozen_b1.py", "utils/bit_supervision.py",
        "utils/semantic_mixup.py", "utils/semantic_rank.py", "utils/rank_checkpoint.py",
        "utils/neighbor_feature_distill.py", "utils/neighborhood_protected_augmentation.py", "utils/foundation_distill.py",
        "utils/neighborhood_view_consistency.py",
    ):
        digest.update(relative.encode())
        # A checkout on Linux/Windows should have the same source identity.
        digest.update((root / relative).read_text(encoding="utf-8").encode())
    return digest.hexdigest()


def build_cache_metadata(args, stage):
    metadata = {
        "stage": stage,
        "dataset": args.dataset,
        "seed": args.seed,
        "split_sha256": args.split_hash,
        "num_classes": args.num_classes,
        "git_commit": args.git_commit,
    }
    if stage in {"classifier", "similarity"}:
        metadata.update(
            {
                "learning_rate": args.lr,
                "classification_epochs": args.classify_epoch,
                "resize_size": args.resize_size,
                "crop_size": args.crop_size,
            }
        )
    if stage in {"classifier", "similarity"} and getattr(args, "paired_stage_seeds", False):
        metadata["classifier_protocol"] = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "validation_interval": args.test_map,
            "deterministic": not args.non_deterministic,
        }
    if stage in {"similarity", "centers"}:
        metadata["mask_strategy"] = args.mask_strategy
    if stage == "centers":
        metadata["similarity_config_hash"] = args.similarity_hash
        metadata["center_update_strategy"] = getattr(
            args, "center_update_strategy", "monotonic_discrete_search"
        )
    if stage in {"mds", "centers"}:
        metadata["code_length"] = args.code_length
    if stage in {"similarity", "centers"} and getattr(args, "variant", "b0") in ("b1", "b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd", "b3rnd", "b3nvc", "b3nmix"):
        from utils.reliable_relation import relation_config
        metadata["relation"] = relation_config(args)
        if stage == "similarity":
            metadata["classifier_sha256"] = args.classifier_sha256
        else:
            metadata["similarity_sha256"] = args.similarity_content_sha256
    metadata["config_hash"] = stable_config_hash(metadata)
    return metadata


def save_cache(path, value, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"value": value, "metadata": metadata}, path)


def load_cache(path, expected_metadata, map_location="cpu"):
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "value" not in payload or "metadata" not in payload:
        raise RuntimeError(f"Legacy cache without metadata is rejected: {path}")
    if payload["metadata"] != expected_metadata:
        raise RuntimeError(
            f"Cache metadata mismatch: {path}\n"
            f"expected={expected_metadata}\nactual={payload['metadata']}"
        )
    return payload["value"]


def write_run_manifest(args):
    os.makedirs(args.run_dir, exist_ok=True)
    manifest = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
    }
    manifest["device"] = str(args.device)
    manifest["torch_version"] = torch.__version__
    manifest["torch_cuda_version"] = torch.version.cuda
    manifest["cuda_available"] = torch.cuda.is_available()
    manifest["cublas_workspace_config"] = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    )
    if torch.device(args.device).type == "cuda" and manifest["cuda_available"]:
        manifest["gpu"] = torch.cuda.get_device_name(args.device)
    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
