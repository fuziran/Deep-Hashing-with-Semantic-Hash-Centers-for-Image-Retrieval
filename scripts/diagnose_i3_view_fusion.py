"""Frozen NFD screen: does two-view binary-code fusion help validation retrieval?

This is a development diagnostic, not an innovation or a final-query result.
It uses the selected 32-bit NFD checkpoint and the original database split.
The final query loader is created for split auditing but is never iterated.
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
from scripts.diagnose_i3_validation import METRICS, checked_run, score_validation
from utils.experiment import file_sha256, seed_everything


def binarize(logits):
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("Expected finite [images, bits] logits")
    return np.where(logits >= 0, 1, -1).astype(np.int8)


@torch.inference_mode()
def encode_views(loader, net, device, bits, description=""):
    original, fused, classes, ids = [], [], [], []
    net.eval()
    for batch_index, (images, labels, indices) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        first = net(images)
        second = net(torch.flip(images, dims=(-1,)))
        if (first.shape != (len(images), bits) or second.shape != first.shape
                or not torch.isfinite(first).all() or not torch.isfinite(second).all()):
            raise ValueError("Invalid frozen student logits")
        original.append(first.cpu().numpy())
        fused.append(((first + second) / 2).cpu().numpy())
        classes.append(labels.argmax(1).numpy())
        ids.append(indices.numpy())
        if description and batch_index % 100 == 0:
            print(f"Encoding {description}: batch {batch_index}/{len(loader)}", flush=True)
    result = dict(original=np.concatenate(original), fused=np.concatenate(fused),
                  classes=np.concatenate(classes), ids=np.concatenate(ids))
    expected_ids = np.asarray(loader.dataset.indices)
    expected_classes = np.asarray(loader.dataset.labels)[expected_ids]
    if (not np.array_equal(result["ids"], expected_ids)
            or not np.array_equal(result["classes"], expected_classes)):
        raise ValueError("Encoded IDs or classes differ from audited split")
    return result


def paired_summary(baseline, candidate, seed=60):
    rng = np.random.default_rng(seed)
    result = {}
    for metric, _ in METRICS:
        old = baseline["ap"][metric]
        new = candidate["ap"][metric]
        delta = new - old
        boot = np.array([delta[rng.integers(len(delta), size=len(delta))].mean()
                         for _ in range(2000)])
        result[metric] = dict(baseline=float(old.mean()), candidate=float(new.mean()),
                              delta=float(delta.mean()),
                              descriptive_paired_bootstrap_95=list(map(
                                  float, np.quantile(boot, [.025, .975]))))
    return result


def analyze(options):
    run = checked_run(options.nfd_run)
    config = run["config"]
    if (config["variant"] != "b2nfd" or config["dataset"] != "cifar-100-new-seg"
            or config["code_length"] != 32 or config["seed"] != 60):
        raise ValueError("Expected selected 32-bit seed-60 B1+NFD run")
    output = Path(options.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --cpu for a small check")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    seed_everything(config["seed"], deterministic=not config.get("non_deterministic", False))
    args = argparse.Namespace(**config)
    args.device = device
    args.num_workers = options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or args.root)).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_i3_view_fusion_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del train, relation, query
        if args.split_hash != config["split_hash"]:
            raise ValueError("Reconstructed split differs from saved run")
        expected_data = config.get("dataset_content_sha256")
        if expected_data and args.dataset_content_sha256 != expected_data:
            raise ValueError("Dataset content differs from saved run")
        report = dict(
            version="i3-frozen-view-fusion-screen-v1",
            protocol="validation vs original database; final query never iterated",
            run_dir=str(run["path"]), selected_epoch=run["summary"]["selected_epoch"],
            checkpoint_sha256=run["checkpoint_sha256"],
            diagnostic_source_sha256=file_sha256(__file__),
            split_hash=args.split_hash,
            dataset_content_sha256=args.dataset_content_sha256,
            code_length=config["code_length"], seed=config["seed"],
            validation_count=len(validation.dataset), database_count=len(database.dataset),
            selected_validation=run["selected"]["validation"], device=str(device),
            view_policy="center crop and horizontal flip; arithmetic mean of raw logits before sign",
            retrieval_rule="unchanged 32-bit Hamming with stable database order")
        if options.dry_run:
            report["dry_run"] = True
            return report
        net = ResNet(config["code_length"], pretrained=False)
        state = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
        net.to(device).eval()
        val = encode_views(validation, net, device, config["code_length"], "validation")
        db = encode_views(database, net, device, config["code_length"], "database")
        del net
        val_original, val_fused = binarize(val["original"]), binarize(val["fused"])
        db_original, db_fused = binarize(db["original"]), binarize(db["fused"])
        scored = {
            "original": score_validation(val_original, db_original, val["classes"],
                                         db["classes"], "original"),
            "query_fusion_only": score_validation(val_fused, db_original, val["classes"],
                                                  db["classes"], "query fusion only"),
            "symmetric_fusion": score_validation(val_fused, db_fused, val["classes"],
                                                 db["classes"], "symmetric fusion"),
        }
        for metric, recorded in run["selected"]["validation"].items():
            if metric in scored["original"]["ap"] and abs(
                    scored["original"]["ap"][metric].mean() - recorded) > 1e-5:
                raise ValueError(f"Original Hamming {metric} does not reproduce saved validation")
        report["changed_code_fraction"] = dict(
            validation=float((val_original != val_fused).any(axis=1).mean()),
            database=float((db_original != db_fused).any(axis=1).mean()))
        report["comparison"] = {
            name: paired_summary(scored["original"], scored[name])
            for name in ("query_fusion_only", "symmetric_fusion")
        }
        report["mean_false_top100"] = {
            name: float(result["false_top100"].mean()) for name, result in scored.items()
        }
        report["note"] = (
            "Exploratory development result on the same validation split that selected the "
            "checkpoint. Bootstrap intervals are descriptive, not independent confirmation. "
            "Two-view fusion doubles encoding work; it is prior art, not a standalone novelty claim. "
            "No model was trained and no final query was scored.")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "per_query_validation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["validation_id", "class_id", "fused_code_changed"] +
                            [f"{name}_{metric}" for name in scored for metric, _ in METRICS])
            for i, image_id in enumerate(val["ids"]):
                writer.writerow([int(image_id), int(val["classes"][i]),
                                 int((val_original[i] != val_fused[i]).any())] +
                                [float(result["ap"][metric][i]) for result in scored.values()
                                 for metric, _ in METRICS])
        (output / "view_fusion.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
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
