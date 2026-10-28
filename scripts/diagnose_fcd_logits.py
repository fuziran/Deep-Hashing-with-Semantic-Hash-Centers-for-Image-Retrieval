"""Compare frozen teacher logits as direct 32-bit retrieval codes.

This is a validation-vs-training-only diagnostic, not a student run or final query
evaluation. It reuses the failed FCD preparation's training logits and head; only
validation DINO features and original-teacher train/validation logits are inferred.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.data_loader import load_data
from scripts.prepare_foundation_teacher import extract
from scripts.run_i3_npa import ARTIFACTS, resolve_file
from utils.experiment import current_git_commit, seed_everything
from utils.foundation_distill import CACHE_VERSION, file_sha256, image_subset_sha256
from utils.frozen_b1 import load_frozen_b1
from utils.neighbor_feature_distill import load_nfd
from utils.reliable_relation import state_sha256
from utils.tools import CalcTopMapPerQuery

CUTOFFS = (-1, 100, 1000)


def center_codes(logits, centers, temperature=None):
    """Use a predicted class center or the softmax-weighted expected center."""
    if logits.ndim != 2 or centers.ndim != 2 or logits.shape[1] != centers.shape[0]:
        raise ValueError("Logits and centers have incompatible shapes")
    if not torch.isfinite(logits).all() or not ((centers == -1) | (centers == 1)).all():
        raise ValueError("Invalid logits or binary centers")
    if temperature is None:
        return centers[logits.argmax(1)].float()
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    scores = F.softmax(logits.float() / temperature, dim=1) @ centers.float()
    return torch.where(scores >= 0, 1.0, -1.0)


@torch.inference_mode()
def old_training_logits(loader, teacher, device):
    rows = []
    teacher.eval()
    for batch, (images, labels, ids) in enumerate(loader):
        images = images.to(device)
        first, _ = teacher.forward_with_features(images)
        second, _ = teacher.forward_with_features(images.flip(-1))
        rows.append((ids.cpu(), labels.argmax(1).cpu(), ((first + second) / 2).cpu()))
        if (batch + 1) % 25 == 0:
            print(f"Original-teacher extraction: batch {batch+1}/{len(loader)}", flush=True)
    values = [torch.cat([row[i] for row in rows]) for i in range(3)]
    order = values[0].argsort()
    return [value[order] for value in values]


def paired_metrics(new_train, new_val, old_train, old_val, train_labels,
                   val_labels, centers, temperature):
    labels_train = F.one_hot(train_labels, centers.shape[0]).numpy()
    labels_val = F.one_hot(val_labels, centers.shape[0]).numpy()

    def per_query(train_logits, val_logits):
        train_codes = center_codes(train_logits, centers, temperature).numpy()
        val_codes = center_codes(val_logits, centers, temperature).numpy()
        return CalcTopMapPerQuery(val_codes, labels_val, train_codes, labels_train, CUTOFFS)

    new = per_query(new_train, new_val)
    old = per_query(old_train, old_val)
    rng = np.random.default_rng(60)
    indices = rng.integers(0, len(val_labels), size=(2000, len(val_labels)))
    report = {}
    for cutoff in CUTOFFS:
        key = str(cutoff)
        delta = np.asarray(new[key]) - np.asarray(old[key])
        low, high = np.quantile(delta[indices].mean(1), [0.025, 0.975])
        report[{ -1: "mAP@ALL", 100: "mAP@100", 1000: "mAP@1000" }[cutoff]] = dict(
            new=float(np.mean(new[key])), old=float(np.mean(old[key])),
            delta=float(np.mean(delta)), paired_bootstrap_95=[float(low), float(high)])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-config", required=True)
    parser.add_argument("--foundation-repo", required=True)
    parser.add_argument("--foundation-weights", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args_cli = parser.parse_args()
    if args_cli.batch_size < 1:
        parser.error("--batch-size must be positive")
    destination = Path(args_cli.output_dir).resolve()
    if destination.exists():
        parser.error("Output directory already exists")
    device = torch.device(f"cuda:{args_cli.gpu}")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    config_path = Path(args_cli.nfd_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("variant") != "b2nfd" or config.get("code_length") != 32
            or config.get("dataset") != "cifar-100-new-seg"):
        parser.error("Expected the paired CIFAR-100 32-bit NFD reference")
    for key in ARTIFACTS:
        config[key] = str(resolve_file(config[key], config_path))
    args = argparse.Namespace(**config)
    args.device = device
    seed_everything(args.seed, deterministic=not args.non_deterministic)
    teacher_dir = Path(args_cli.teacher_dir).resolve()
    payload = torch.load(teacher_dir / "foundation_cache.pt", map_location="cpu", weights_only=False)
    identity = payload["identity"]
    if (identity.get("version") != CACHE_VERSION or identity.get("dataset") != args.dataset
            or identity.get("seed") != args.seed or identity.get("split_hash") != args.split_hash):
        raise ValueError("Foundation cache does not match the NFD split")
    with tempfile.TemporaryDirectory(prefix="shc_fcd_diagnostic_") as scratch:
        args.output_dir = scratch
        train, relation, val, _, _, *_ = load_data(args)
        if args.split_hash != config["split_hash"]:
            raise ValueError("Reconstructed split differs from the reference")
        if identity["training_images_sha256"] != image_subset_sha256(train.dataset):
            raise ValueError("Training images differ from the foundation cache")
        similarity, centers = load_frozen_b1(args)
        if centers.shape != (args.num_classes, 32):
            centers = centers.T
        if (identity["classifier_sha256"] != args.classifier_sha256
                or not torch.equal(payload["ids"], torch.as_tensor(np.sort(train.dataset.indices)))):
            raise ValueError("Foundation cache does not match B1 classifier or training IDs")
        expected_labels = torch.as_tensor(np.asarray(train.dataset.labels)[payload["ids"].numpy()]).long()
        if not torch.equal(payload["labels"], expected_labels):
            raise ValueError("Foundation cache labels differ from the training split")
        del similarity
        nfd = load_nfd(args)
        repo = Path(args_cli.foundation_repo).resolve()
        weights = Path(args_cli.foundation_weights).resolve()
        if (not (repo / "hubconf.py").is_file()
                or file_sha256(weights) != identity["foundation_weights_sha256"]):
            raise ValueError("DINOv3 repository or weights differ from teacher preparation")
        foundation = torch.hub.load(str(repo), "dinov3_vitb16", source="local", pretrained=False)
        foundation.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        foundation = foundation.eval().requires_grad_(False).to(device)
        state = torch.load(teacher_dir / "foundation_classifier.pt", map_location="cpu", weights_only=True)
        if state_sha256(state) != identity["teacher_head_sha256"]:
            raise ValueError("Foundation classifier differs from teacher preparation")
        head = nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
        head.load_state_dict(state, strict=True)
        head = head.eval().requires_grad_(False).to(device)
        loader_args = dict(batch_size=args_cli.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
        relation_loader = DataLoader(relation.dataset, **loader_args)
        validation_loader = DataLoader(val.dataset, **loader_args)
        print("Inferring original-teacher training logits; DINO training logits are reused", flush=True)
        old_ids, old_labels, old_train = old_training_logits(relation_loader, nfd.teacher, device)
        if not torch.equal(old_ids, payload["ids"]) or not torch.equal(old_labels, payload["labels"]):
            raise ValueError("Training IDs or labels disagree between teachers")
        print("Inferring DINOv3 and original-teacher validation logits", flush=True)
        val_ids, val_labels, new_features, _, old_val = extract(
            validation_loader, foundation, nfd.teacher, device)
        if not torch.equal(val_ids, torch.as_tensor(np.sort(val.dataset.indices))):
            raise ValueError("Validation IDs disagree with the split")
        with torch.inference_mode():
            new_val = head(new_features.to(device)).cpu()
        check = payload["teacher_check"]
        for name, logits, expected in (("new", new_val, check["new_accuracy"]),
                                       ("old", old_val, check["old_accuracy"])):
            actual = (logits.argmax(1) == val_labels).float().mean().item()
            if abs(actual - expected) > 1e-6:
                raise ValueError(f"{name} validation accuracy differs from teacher preparation")
        report = dict(
            protocol="validation vs training-only gallery; direct class-to-B1-center codes",
            interpretation="exploratory diagnostic only; no student or final query evaluation",
            nfd_config=str(config_path), teacher_dir=str(teacher_dir),
            teacher_check_passed=check["passed"],
            teacher_hash_probe=dict(new=check["new_hash_probe"], old=check["old_hash_probe"]),
            foundation_weights_sha256=identity["foundation_weights_sha256"],
            diagnostic_source_sha256=file_sha256(__file__), git_commit=current_git_commit(),
            counts=dict(train=len(old_ids), validation=len(val_ids)),
            hard_center=paired_metrics(payload["logits"], new_val, old_train, old_val,
                                       old_labels, val_labels, centers, None),
            soft_center_T2=paired_metrics(payload["logits"], new_val, old_train, old_val,
                                         old_labels, val_labels, centers, 2.0),
        )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
