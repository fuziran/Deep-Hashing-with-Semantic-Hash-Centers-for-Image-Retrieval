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


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_metadata(args, stage):
    metadata = {
        "stage": stage,
        "protocol": getattr(args, "protocol", "audited_b0"),
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
    if stage in {"similarity", "centers"}:
        metadata["mask_strategy"] = args.mask_strategy
    if stage == "centers":
        metadata["similarity_config_hash"] = args.similarity_hash
        metadata["center_update_strategy"] = getattr(
            args, "center_update_strategy", "monotonic_discrete_search"
        )
    if stage in {"mds", "centers"}:
        metadata["code_length"] = args.code_length
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
    local_weight_path = "./models_ckpt/resnet34-b627a593.pth"
    if os.path.exists(local_weight_path):
        manifest["pretrained_weights"] = {
            "path": os.path.abspath(local_weight_path),
            "sha256": file_sha256(local_weight_path),
        }
    else:
        manifest["pretrained_weights"] = (
            "torchvision.models.ResNet34_Weights.IMAGENET1K_V1"
        )
    if torch.cuda.is_available():
        manifest["gpu"] = torch.cuda.get_device_name(args.device)
    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
