"""Verify and reuse NABirds-B B1 relation and center artifacts for I2-NFD."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from utils.experiment import build_cache_metadata, stable_config_hash
from utils.reliable_relation import relation_config, tensor_sha256, validate_similarity


def load_frozen_b1(args):
    config_path = Path(args.parent_b1_config)
    parent = json.loads(config_path.read_text(encoding="utf-8"))
    if parent.get("variant") != "b1" or parent.get("dataset", "").lower() != "nabirds-b":
        raise ValueError("I2-NFD requires a NABirds-B B1 parent run")
    if parent.get("paired_stage_seeds", False) != args.paired_stage_seeds:
        raise ValueError("B1 parent and I2-NFD must use the same paired-stage-seeds setting")
    fields = (
        "dataset", "root", "seed", "split_hash", "num_classes", "code_length",
        "lr", "epoch", "classify_epoch", "test_map", "beta", "lambd", "topK",
        "batch_size", "num_workers", "resize_size", "crop_size", "val_per_class",
        "mask_strategy", "center_update_strategy", "non_deterministic",
        "relation_mode", "relation_kappa", "relation_temperature", "relation_weight_floor",
    )
    for key in fields:
        if key not in parent or parent[key] != getattr(args, key):
            raise ValueError(f"B1 parent configuration mismatch: {key}")

    def read_cache(path):
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or not {"value", "metadata"} <= payload.keys():
            raise ValueError(f"Invalid audited cache: {path}")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid cache metadata: {path}")
        identity = dict(metadata)
        if identity.pop("config_hash", None) != stable_config_hash(identity):
            raise ValueError(f"Invalid metadata hash: {path}")
        return payload["value"], metadata

    parent_args = argparse.Namespace(**parent)
    artifact, relation_meta = read_cache(args.similarity_cache)
    if not isinstance(artifact, dict):
        raise ValueError("I2-NFD requires an I1 relation artifact")
    similarity = artifact["similarity"]
    validate_similarity(similarity, args.num_classes)
    digest = tensor_sha256(similarity)
    if digest != parent["similarity_content_sha256"] or digest != artifact["similarity_sha256"]:
        raise ValueError("Parent relation content mismatch")
    if artifact["relation_config"] != relation_config(args):
        raise ValueError("Parent relation settings mismatch")
    args.classifier_sha256 = parent["classifier_sha256"]
    if artifact["classifier_sha256"] != args.classifier_sha256:
        raise ValueError("Parent classifier provenance mismatch")
    args.similarity_hash = parent["similarity_hash"]
    args.similarity_content_sha256 = digest

    centers, center_meta = read_cache(args.center_cache)
    if not isinstance(centers, torch.Tensor) or centers.shape not in (
        (args.num_classes, args.code_length), (args.code_length, args.num_classes)
    ):
        raise ValueError("Parent center shape mismatch")
    if not ((centers == -1) | (centers == 1)).all():
        raise ValueError("Parent centers must contain only -1/+1")
    if tensor_sha256(centers) != parent["center_content_sha256"]:
        raise ValueError("Parent center content mismatch")

    for stage, actual in (("similarity", relation_meta), ("centers", center_meta)):
        expected = build_cache_metadata(parent_args, stage)
        if actual != expected:
            raise ValueError(f"Parent {stage} cache metadata mismatch")
        if actual["git_commit"] != parent["git_commit"]:
            raise ValueError(f"Parent {stage} commit mismatch")
    if relation_meta["config_hash"] != args.similarity_hash:
        raise ValueError("Parent relation cache identity mismatch")

    args.parent_b1_run_id = parent["run_id"]
    args.parent_b1_git_commit = parent["git_commit"]
    args.parent_b1_source_sha256 = parent["source_sha256"]
    args.parent_b1_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    args.parent_center_config_hash = center_meta["config_hash"]
    args.center_content_sha256 = tensor_sha256(centers)
    return similarity, centers
