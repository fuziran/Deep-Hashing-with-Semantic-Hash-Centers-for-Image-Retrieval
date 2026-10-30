"""Fit three fixed hash readouts on training-only frozen NFD features.

This pilot evaluates real validation retrieval. The final query is never
iterated. All fitted heads share the same selected NFD backbone and 32 bits.
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

from data.data_loader import load_data, _resolve_cifar100_root
from network import ResNet
from scripts.diagnose_i3_bottleneck import encode
from scripts.diagnose_i3_neighbor_fusion import load_nfd_geometry
from scripts.diagnose_i3_validation import METRICS, checked_run, score_validation
from scripts.diagnose_i3_view_fusion import paired_summary
from utils.experiment import file_sha256, seed_everything
from utils.frozen_b1 import load_frozen_b1
from utils.semantic_readout import VERSION, RIDGE_MULTIPLIER, GEOMETRY_WEIGHT, fit_readout, readout_logits


def readout_codes(features, head, chunk_size=4096):
    parts = []
    for start in range(0, len(features), chunk_size):
        logits = readout_logits(features[start:start + chunk_size], head)
        parts.append(torch.where(logits >= 0, 1, -1).to(torch.int8).numpy())
    return np.concatenate(parts)


def analyze(options):
    run = checked_run(options.nfd_run)
    config = run["config"]
    if (config["variant"] != "b2nfd" or config["code_length"] != 32
            or config["seed"] != 60 or config.get("nfd_alpha", 0) <= 0
            or config.get("nfd_weight", 0) <= 0):
        raise ValueError("Expected paired 32-bit seed-60 B1+NFD reference")
    output = Path(options.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --cpu for synthetic checks")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    seed_everything(config["seed"], deterministic=not config.get("non_deterministic", False))
    args = argparse.Namespace(**config)
    args.device, args.num_workers = device, options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or args.root)).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_i3_readout_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del train, query
        if args.split_hash != config["split_hash"]:
            raise ValueError("Reconstructed split differs from NFD training")
        expected_data = config.get("dataset_content_sha256")
        if expected_data and args.dataset_content_sha256 != expected_data:
            raise ValueError("Dataset content differs from NFD training")
        report = dict(
            version=VERSION, reference_run=str(run["path"]),
            checkpoint_sha256=run["checkpoint_sha256"],
            selected_backbone_epoch=run["summary"]["selected_epoch"],
            diagnostic_source_sha256=file_sha256(__file__),
            readout_source_sha256=file_sha256(ROOT / "utils" / "semantic_readout.py"),
            split_hash=args.split_hash, dataset_content_sha256=args.dataset_content_sha256,
            training_count=len(relation.dataset), validation_count=len(validation.dataset),
            database_count=len(database.dataset), code_length=config["code_length"],
            device=str(device), query_evaluated=False,
            protocol="heads fit on training only; validation vs original database; final query never iterated",
            fixed_hyperparameters=dict(ridge_multiplier=RIDGE_MULTIPLIER,
                                       geometry_weight=GEOMETRY_WEIGHT),
            inference="one student view; normalized features plus linear head; 32-bit Hamming")
        if options.dry_run:
            report["dry_run"] = True
            return report
        args.num_workers = config["num_workers"]
        _, centers = load_frozen_b1(args)
        if centers.shape == (config["code_length"], config["num_classes"]):
            centers = centers.T
        if centers.shape != (config["num_classes"], config["code_length"]):
            raise ValueError("Invalid frozen B1 centers")
        prototypes, neighbors, prototype_hash = load_nfd_geometry(
            config, run["path"], torch.device("cpu"))
        report["prototype_sha256"] = prototype_hash
        report["nfd_basis_sha256"] = config["nfd_basis_sha256"]
        net = ResNet(config["code_length"], pretrained=False)
        net.load_state_dict(torch.load(run["checkpoint"], map_location="cpu",
                                       weights_only=True), strict=True)
        net.to(device).eval().requires_grad_(False)
        training = encode(relation, net, device, config["code_length"], description="training features")
        val = encode(validation, net, device, config["code_length"], description="validation features")
        db = encode(database, net, device, config["code_length"], description="database features")
        del net
        scored = {"original": score_validation(val["codes"], db["codes"],
                                                 val["classes"], db["classes"], "original head")}
        for metric, recorded in run["selected"]["validation"].items():
            if metric in scored["original"]["ap"] and abs(
                    scored["original"]["ap"][metric].mean() - recorded) > 1e-5:
                raise ValueError(f"Original {metric} differs from selected validation")
        heads = {}
        for name, geometry in (("ridge", "none"), ("global_geometry", "global"),
                                ("nfd_geometry", "nfd")):
            print(f"Fitting fixed {name} readout from {len(training['ids'])} training features", flush=True)
            head = fit_readout(training["features"], training["classes"],
                               centers, prototypes, neighbors, geometry)
            val_codes, db_codes = readout_codes(val["features"], head), readout_codes(db["features"], head)
            scored[name] = score_validation(val_codes, db_codes, val["classes"], db["classes"], name)
            heads[name] = head
        report["results"] = {
            name: {**{metric: float(scored[name]["ap"][metric].mean()) for metric, _ in METRICS},
                   "mean_false_top100": float(scored[name]["false_top100"].mean())}
            for name in scored
        }
        report["head_fits"] = {name: head["metadata"] for name, head in heads.items()}
        report["comparison_to_original"] = {
            name: paired_summary(scored["original"], scored[name]) for name in heads
        }
        report["nfd_geometry_vs_ridge"] = paired_summary(scored["ridge"], scored["nfd_geometry"])
        report["nfd_geometry_vs_global"] = paired_summary(scored["global_geometry"], scored["nfd_geometry"])
        report["note"] = (
            "Fixed-budget exploratory readout fit after backbone selection. Validation and database "
            "labels are used only in evaluation, never in head fitting. The same validation split "
            "selected the backbone, so bootstrap intervals are descriptive. This is additional "
            "supervised calibration on the original training subset, not another 300-epoch run. "
            "Analytical hashing and relational supervision have prior art; this pilot proves no novelty. "
            "No final query was scored.")
        output.mkdir(parents=True, exist_ok=False)
        torch.save(dict(version=VERSION, heads=heads,
                        provenance={key: report[key] for key in
                                    ("checkpoint_sha256", "split_hash", "dataset_content_sha256",
                                     "prototype_sha256", "readout_source_sha256")}),
                   output / "readout_heads.pt")
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method"] + list(report["results"]["original"]))
            writer.writeheader()
            for name, values in report["results"].items():
                writer.writerow(dict(method=name, **values))
        with (output / "per_query_validation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["validation_id", "class_id"] +
                            [f"{name}_{metric}" for name in scored for metric, _ in METRICS])
            for index, image_id in enumerate(val["ids"]):
                writer.writerow([int(image_id), int(val["classes"][index])] +
                                [float(values["ap"][metric][index])
                                 for values in scored.values() for metric, _ in METRICS])
        (output / "readout_screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-run", required=True)
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
