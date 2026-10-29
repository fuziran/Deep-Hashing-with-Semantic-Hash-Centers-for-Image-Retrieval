"""Read-only feasibility audit for the I3 reliable-neighborhood gate.

The frozen I1 teacher sees one training augmentation and its horizontal flip,
then deterministic validation crops and their flips. The final query and original
database loaders are constructed for split verification but never iterated.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from data.data_loader import load_data, _resolve_cifar100_root
from utils.experiment import file_sha256, seed_everything
from utils.neighbor_feature_distill import load_nfd, RELIABLE_VERSION


@torch.inference_mode()
def inspect(loader, teacher, device, description):
    teacher.eval()
    rows = []
    for batch_index, (images, labels, ids) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        target = labels.argmax(1).to(device)
        features = teacher.forward_features(images)
        flipped_features = teacher.forward_features(torch.flip(images, dims=(-1,)))
        logits = teacher.classify_layer(features)
        flipped_logits = teacher.classify_layer(flipped_features)
        if (logits.shape != (len(images), labels.shape[1])
                or flipped_logits.shape != logits.shape
                or not torch.isfinite(logits).all()
                or not torch.isfinite(flipped_logits).all()):
            raise ValueError("Invalid frozen teacher predictions")
        correct = logits.argmax(1) == target
        flip_correct = flipped_logits.argmax(1) == target
        gate = correct & flip_correct
        margin = logits.gather(1, target[:, None]).squeeze(1) - logits.masked_fill(
            F.one_hot(target, logits.shape[1]).bool(), -torch.inf).amax(1)
        flip_margin = flipped_logits.gather(1, target[:, None]).squeeze(1) - flipped_logits.masked_fill(
            F.one_hot(target, logits.shape[1]).bool(), -torch.inf).amax(1)
        for image_id, class_id, right, flip_right, trusted, first_margin, second_margin in zip(
                ids.tolist(), target.cpu().tolist(), correct.cpu().tolist(),
                flip_correct.cpu().tolist(), gate.cpu().tolist(), margin.cpu().tolist(),
                flip_margin.cpu().tolist()):
            rows.append(dict(image_id=image_id, class_id=class_id,
                             correct=right, flip_correct=flip_right, gate=trusted,
                             margin=first_margin, flip_margin=second_margin))
        if batch_index % 50 == 0:
            print(f"{description}: batch {batch_index}/{len(loader)}", flush=True)
    expected_ids = np.asarray(loader.dataset.indices)
    expected_classes = np.asarray(loader.dataset.labels)[expected_ids]
    if (len(rows) != len(expected_ids) or len(set(row["image_id"] for row in rows)) != len(rows)
            or set(row["image_id"] for row in rows) != set(map(int, expected_ids))
            or any(row["class_id"] != int(loader.dataset.labels[row["image_id"]]) for row in rows)
            or sorted(row["class_id"] for row in rows) != sorted(map(int, expected_classes))):
        raise ValueError("Teacher predictions do not cover the audited image IDs/classes")
    return rows


def summarize(rows):
    classes = sorted(set(row["class_id"] for row in rows))
    class_gate = {str(c): sum(row["gate"] for row in rows if row["class_id"] == c) /
                  sum(row["class_id"] == c for row in rows) for c in classes}
    return dict(count=len(rows), original_accuracy=sum(row["correct"] for row in rows) / len(rows),
                flipped_accuracy=sum(row["flip_correct"] for row in rows) / len(rows),
                gate_fraction=sum(row["gate"] for row in rows) / len(rows),
                class_gate_min=min(class_gate.values()), class_gate_max=max(class_gate.values()),
                class_gate_fraction=class_gate)


def compare_validation(rows, audit_root, reference_config, reference_dir):
    by_id = {row["image_id"]: row for row in rows}
    ap = {}
    for name in ("nfd", "ordinary_fd"):
        audit = json.loads((Path(audit_root) / name / "diagnostic.json").read_text(encoding="utf-8"))
        if (audit.get("split_hash") != reference_config["split_hash"]
                or audit.get("dataset_content_sha256") != reference_config["dataset_content_sha256"]
                or audit.get("code_length") != reference_config["code_length"]
                or audit.get("seed") != reference_config["seed"]):
            raise ValueError("Frozen-student audit identity differs from the NFD reference")
        if name == "nfd" and Path(audit["run_dir"]).resolve() != Path(reference_dir).resolve():
            raise ValueError("Frozen NFD audit points to a different reference run")
        audited_config = json.loads((Path(audit["run_dir"]) / "config.json").read_text(encoding="utf-8"))
        if (audited_config.get("variant") != "b2nfd"
                or (name == "nfd" and audited_config.get("nfd_alpha") != reference_config["nfd_alpha"])
                or (name == "ordinary_fd" and audited_config.get("nfd_alpha") != 0)):
            raise ValueError("Frozen-student audit is not the intended NFD/ordinary-FD pair")
        path = Path(audit_root) / name / "per_query_validation.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            ap[name] = {int(row["validation_id"]): row for row in csv.DictReader(stream)}
    if set(by_id) != set(ap["nfd"]) or set(by_id) != set(ap["ordinary_fd"]):
        raise ValueError("Reliability audit and frozen-student audit have different validation IDs")
    result = {}
    for gate in (True, False):
        ids = [image_id for image_id, row in by_id.items() if row["gate"] == gate]
        result["trusted" if gate else "untrusted"] = dict(
            count=len(ids),
            nfd_minus_ordinary_binary_mAP_ALL=(sum(
                float(ap["nfd"][i]["binary_hamming_mAP@ALL"])
                - float(ap["ordinary_fd"][i]["binary_hamming_mAP@ALL"])
                for i in ids) / len(ids) if ids else None),
        )
    return result


def analyze(options):
    config_path = Path(options.nfd_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("variant") != "b2nfd" or config.get("nfd_alpha", 0) <= 0
            or config.get("seed") != 60 or config.get("code_length") != 32
            or not config.get("paired_stage_seeds")):
        raise ValueError("Expected the paired 32-bit seed-60 NFD reference")
    output = Path(options.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --cpu only for a small check")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    seed_everything(config["seed"], deterministic=not config.get("non_deterministic", False))
    args = argparse.Namespace(**config)
    args.device = device
    args.num_workers = options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or args.root)).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_rgnfd_gate_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del relation, query, database
        if args.split_hash != config["split_hash"]:
            raise ValueError("Reconstructed split differs from NFD reference")
        expected_data = config.get("dataset_content_sha256")
        if expected_data and expected_data != args.dataset_content_sha256:
            raise ValueError("Dataset pixels/labels differ from NFD reference")
        report = dict(version=RELIABLE_VERSION,
                      protocol="training augmentation and validation center crop; final query never iterated",
                      reference_run=str(config_path.parent), teacher_cache=str(config["teacher_cache"]),
                      split_hash=args.split_hash, dataset_content_sha256=args.dataset_content_sha256,
                      diagnostic_source_sha256=file_sha256(__file__), device=str(device))
        if options.dry_run:
            report["dry_run"] = True
            return report
        nfd = load_nfd(args)
        report["teacher_sha256"] = args.nfd_teacher_sha256
        report["projection_sha256"] = args.nfd_basis_sha256
        training = inspect(train, nfd.teacher, device, "training-only gate")
        heldout = inspect(validation, nfd.teacher, device, "validation gate")
        report["training"] = summarize(training)
        report["validation"] = summarize(heldout)
        report["training_gate_has_variation"] = 0.05 <= report["training"]["gate_fraction"] <= 0.95
        if options.audit_root:
            report["validation_group_delta"] = compare_validation(
                heldout, options.audit_root, config, config_path.parent)
        report["note"] = ("Exploratory feasibility only. The actual training gate is computed online on "
                          "each augmented batch, so these one-pass rates may differ. Validation groups "
                          "are descriptive, not independent evidence or a threshold-tuning set.")
        output.mkdir(parents=True, exist_ok=False)
        for name, rows in (("training_gate.csv", training), ("validation_gate.csv", heldout)):
            with (output / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (output / "gate_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-config", required=True)
    parser.add_argument("--audit-root", help="Existing ordinary-FD/NFD bottleneck audit root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if options.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    try:
        report = analyze(options)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
