"""Training-only feasibility screen for NFD neighborhood view consistency.

The selected NFD student and frozen I1 teacher are read only. Validation,
final query and database loaders are constructed for split identity but never
iterated. This report is exploratory and cannot establish retrieval gain.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from data.data_loader import load_data, _resolve_cifar100_root
from network import ResNet
from utils.experiment import file_sha256, seed_everything
from utils.neighbor_feature_distill import load_nfd
from utils.neighborhood_view_consistency import VERSION, semantic_weight, consistency_loss


@torch.inference_mode()
def inspect(loader, student, nfd, centers, device, temperature, margin):
    student.eval()
    nfd.teacher.eval()
    changed, risk, gates, ratios = [], [], [], []
    correct_weak = correct_strong = 0
    observed = set()
    for batch_index, (weak, labels, ids, strong) in enumerate(loader, 1):
        if observed.intersection(ids.tolist()):
            raise ValueError("Duplicate training image ID")
        observed.update(ids.tolist())
        weak, strong = weak.to(device), strong.to(device)
        targets = labels.argmax(1).to(device)
        weak_logits = student(weak)
        strong_logits = student(strong)
        weak_teacher = nfd.teacher.forward_features(weak)
        strong_teacher = nfd.teacher.forward_features(strong)
        correct_weak += int((nfd.teacher.classify_layer(weak_teacher).argmax(1)
                             == targets).sum())
        correct_strong += int((nfd.teacher.classify_layer(strong_teacher).argmax(1)
                               == targets).sum())
        gate, ratio = semantic_weight(weak_teacher, strong_teacher,
                                      nfd.basis, nfd.ranks, targets, temperature)
        _, stats = consistency_loss(weak_logits, strong_logits, targets,
                                     centers, gate, margin)
        changed.extend(((weak_logits >= 0) != (strong_logits >= 0))
                       .any(1).cpu().tolist())
        risk.append(stats["nvc_risk_bit_fraction"] * len(ids))
        gates.extend(gate.cpu().tolist())
        ratios.extend(ratio.cpu().tolist())
        if batch_index % 50 == 0:
            print(f"Training-only views: batch {batch_index}/{len(loader)}", flush=True)
    expected = set(map(int, loader.dataset.indices))
    if observed != expected or len(gates) != len(expected):
        raise ValueError("Training-only preflight did not cover exact image IDs")
    values = torch.tensor(gates)
    return dict(count=len(gates), changed_image_fraction=sum(changed) / len(changed),
                risk_bit_fraction=sum(risk) / len(gates),
                teacher_weak_accuracy=correct_weak / len(gates),
                teacher_strong_accuracy=correct_strong / len(gates),
                gate_mean=float(values.mean()), gate_p10=float(torch.quantile(values, .1)),
                gate_p90=float(torch.quantile(values, .9)),
                projected_ratio_mean=sum(ratios) / len(ratios))


def analyze(options):
    config_path = Path(options.nfd_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("variant") != "b2nfd" or config.get("dataset") != "cifar-100-new-seg"
            or config.get("seed") != 60 or config.get("code_length") != 32
            or config.get("nfd_alpha", 0) <= 0 or config.get("nfd_weight", 0) <= 0
            or not config.get("paired_stage_seeds")):
        raise ValueError("Expected paired 32-bit seed-60 B1+NFD reference")
    output = Path(options.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = config_path.parent / "best_model_state.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --cpu only for a small check")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    seed_everything(config["seed"], deterministic=not config.get("non_deterministic", False))
    args = argparse.Namespace(**config)
    args.variant = "b3nvc"
    args.device = device
    args.num_workers = options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or args.root)).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_nvc_preflight_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del relation, validation, query, database
        if args.split_hash != config["split_hash"]:
            raise ValueError("Split differs from NFD reference")
        if (config.get("dataset_content_sha256")
                and args.dataset_content_sha256 != config["dataset_content_sha256"]):
            raise ValueError("Dataset content differs from NFD reference")
        report = dict(version=VERSION, reference_run=str(config_path.parent),
                      reference_config_sha256=file_sha256(config_path),
                      reference_checkpoint_sha256=file_sha256(checkpoint),
                      nfd_alpha=config["nfd_alpha"], nfd_weight=config["nfd_weight"],
                      split_hash=args.split_hash,
                      dataset_content_sha256=args.dataset_content_sha256,
                      device=str(device), temperature=options.temperature,
                      margin=options.margin,
                      protocol="training-only paired views; final query/database never iterated")
        if options.dry_run:
            report["dry_run"] = True
            return report
        nfd = load_nfd(args)
        report["teacher_sha256"] = args.nfd_teacher_sha256
        report["projection_sha256"] = args.nfd_basis_sha256
        student = ResNet(config["code_length"], pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        student.load_state_dict(state, strict=True)
        student.to(device).eval()
        # Cache formats vary; use the same audited frozen-B1 loader as training.
        from utils.frozen_b1 import load_frozen_b1
        args.num_workers = config["num_workers"]
        _, centers = load_frozen_b1(args)
        if centers.shape == (config["code_length"], config["num_classes"]):
            centers = centers.T
        if centers.shape != (config["num_classes"], config["code_length"]):
            raise ValueError("Frozen B1 centers have the wrong class/bit shape")
        centers = centers.to(device)
        report["training"] = inspect(train, student, nfd, centers, device,
                                      options.temperature, options.margin)
        t = report["training"]
        report["passed"] = bool(
            t["changed_image_fraction"] >= .02
            and t["risk_bit_fraction"] >= .005
            and t["gate_p90"] - t["gate_p10"] >= .05
            and t["teacher_strong_accuracy"] >= .6)
        report["screen_rules"] = dict(min_changed_image_fraction=.02,
                                      min_risk_bit_fraction=.005,
                                      min_gate_p90_minus_p10=.05,
                                      min_teacher_strong_accuracy=.6)
        report["note"] = "Exploratory feasibility, not retrieval efficacy. No validation or final query was scored."
        output.mkdir(parents=True, exist_ok=False)
        (output / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=.05)
    parser.add_argument("--margin", type=float, default=.5)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if (options.num_workers < 0 or not math.isfinite(options.temperature)
            or options.temperature <= 0 or not math.isfinite(options.margin)
            or options.margin <= 0):
        parser.error("Invalid workers, temperature or margin")
    try:
        report = analyze(options)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
