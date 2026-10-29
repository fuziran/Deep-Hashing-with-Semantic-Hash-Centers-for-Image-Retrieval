"""Evaluate a completed paired NFD/FCD selection without teachers or training.

Only new runs with selection receipts are accepted. Use the training source
checkout; no source/hash override or alternative epoch/checkpoint is supported.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from data.data_loader import load_data, _resolve_cifar100_root
from network import ResNet
from utils.experiment import (current_git_commit, file_sha256, seed_everything,
                              selection_receipt, source_sha256)
from utils.tools import CalcTopMapPerQuery, CalcTopMapWithPR, compute_result

METRICS = ("mAP@ALL", "mAP@100", "mAP@1000")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_run(run_dir):
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "metrics.json")
    history = read_json(run_dir / "training_history.json")
    if (config.get("variant") not in ("b2nfd", "b3fcd", "b3rnd", "b3nvc", "b3nmix")
            or config.get("dataset") != "cifar-100-new-seg"
            or config.get("num_classes") != 100
            or config.get("paired_stage_seeds") is not True
            or config.get("topK") != [-1, 100, 1000]):
        raise ValueError("Expected a paired CIFAR-100 NFD/FCD run with original cutoffs")
    if config.get("source_sha256") != source_sha256():
        raise ValueError("Training source SHA256 mismatch; use the exact training checkout")
    receipt = selection_receipt(run_dir)
    if summary.get("selection_receipt_version") != 1:
        raise ValueError("Missing selection receipt; historical runs need their original evaluation workflow")
    for key, actual in receipt.items():
        if actual is None or summary.get(key) != actual:
            raise ValueError(f"Selection receipt mismatch: {key}")
    if (not isinstance(history, list) or len(history) != config["epoch"]
            or [row.get("epoch") for row in history] != list(range(1, config["epoch"] + 1))):
        raise ValueError("Training history is incomplete or epochs are not consecutive")
    validated = [row for row in history if "validation" in row]
    if not validated or any(
            not isinstance(row["validation"].get(key), (int, float))
            or not math.isfinite(row["validation"][key]) for row in validated for key in METRICS):
        raise ValueError("Invalid validation history")
    # max preserves the first winner on ties, exactly as train.py's strict >.
    best = max(validated, key=lambda row: row["validation"]["mAP@ALL"])
    if (summary.get("selection_metric") != "validation mAP@ALL"
            or summary.get("selected_epoch") != best["epoch"]
            or summary.get("best_validation_mAP@ALL") != best["validation"]["mAP@ALL"]):
        raise ValueError("Selected epoch/score does not match validation mAP@ALL history")
    if not config.get("dataset_content_sha256"):
        raise ValueError("Missing training dataset content SHA256")
    return config, summary, best["validation"], receipt


def evaluate(options):
    run_dir = Path(options.run_dir).resolve()
    config, training, validation, receipt = validate_run(run_dir)
    metrics_digest = file_sha256(run_dir / "metrics.json")
    destination = Path(options.output_dir).resolve() if options.output_dir else run_dir / "final_query"
    if destination.exists():
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; --cpu is available for diagnostics")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    args = argparse.Namespace(**config)
    args.device = device
    # Explicit overrides must contain the actual raw files; do not silently fall back.
    if options.data_root:
        candidate = Path(options.data_root).resolve()
        if not all((candidate / name).is_file() for name in ("train", "test", "meta")):
            raise ValueError("--data-root must contain CIFAR-100 train/test/meta")
        args.root = str(candidate)
    args.root = str(Path(_resolve_cifar100_root(args.root)).resolve())
    seed_everything(args.seed, deterministic=not args.non_deterministic)
    checkpoint = run_dir / "best_model_state.pt"
    # No ImageNet/DINO weights, teacher caches or optimizer are constructed.
    net = ResNet(args.code_length, pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                   for value in state.values())):
        raise ValueError("Checkpoint must be a finite student state_dict")
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    with tempfile.TemporaryDirectory(prefix="shc_query_") as scratch:
        # load_data only constructs loaders; training/relation/validation are not iterated.
        args.output_dir = scratch
        _, _, _, query, database, _, _, num_query, num_database = load_data(args)
        if args.split_hash != config["split_hash"]:
            raise ValueError("Reconstructed data split SHA256 mismatch")
        if args.dataset_content_sha256 != config["dataset_content_sha256"]:
            raise ValueError("Dataset pixels/labels SHA256 mismatch")
        if num_query != len(query.dataset) or num_database != len(database.dataset):
            raise ValueError("Evaluation loader counts differ from the manifest")
        report = dict(
            run_dir=str(run_dir), run_id=config["run_id"], variant=config["variant"],
            code_length=config["code_length"], seed=config["seed"],
            fcd_mode=config.get("fcd_mode"),
            fcd_exploratory=config.get("fcd_exploratory", False),
            fcd_original_teacher_check_passed=config.get("fcd_original_teacher_check_passed"),
            fcd_diagnostic_sha256=config.get("fcd_diagnostic_sha256"),
            selected_epoch=training["selected_epoch"],
            selection_metric=training["selection_metric"], selected_validation=validation,
            split_hash=args.split_hash, dataset_content_sha256=args.dataset_content_sha256,
            training_source_sha256=config["source_sha256"],
            evaluation_source_sha256=source_sha256(), evaluation_git_commit=current_git_commit(),
            evaluator_sha256=file_sha256(__file__), training_metrics_sha256=metrics_digest,
            device=str(device), torch_version=str(torch.__version__),
            torch_cuda_version=torch.version.cuda, data_root=args.root,
            batch_size=args.batch_size, num_workers=args.num_workers,
            counts=dict(query=num_query, database=num_database), **receipt,
        )
        if options.dry_run:
            report.update(dry_run=True, query_evaluated=False)
            return report
        destination.mkdir(parents=True, exist_ok=False)
        split_dir = Path(scratch) / "splits"
        if split_dir.is_dir():
            shutil.copytree(split_dir, destination / "splits")
        # Write the request first; metrics.json is written only after full success.
        (destination / "evaluation_request.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        query_binary, query_label = compute_result(query, net, device)
        database_binary, database_label = compute_result(database, net, device)
        for codes, labels, loader in ((query_binary, query_label, query),
                                      (database_binary, database_label, database)):
            indices = np.asarray(loader.dataset.indices)
            expected = np.asarray(loader.dataset.labels)[indices]
            if (codes.shape != (len(indices), args.code_length)
                    or not ((codes == -1) | (codes == 1)).all()
                    or labels.shape != (len(indices), args.num_classes)
                    or not np.array_equal(labels.argmax(1).numpy(), expected)):
                raise ValueError("Retrieved codes/labels do not match dataset IDs")
        arrays = [value.numpy() for value in (query_binary, query_label, database_binary, database_label)]
        values, pr_data = CalcTopMapWithPR(*arrays, args.topK, num_database)
        per_query = CalcTopMapPerQuery(*arrays, args.topK)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Non-finite query metrics")
        if selection_receipt(run_dir) != receipt or file_sha256(run_dir / "metrics.json") != metrics_digest:
            raise ValueError("Training artifacts changed during query evaluation")
        torch.save(dict(
            query_binary=query_binary, query_label=query_label,
            database_binary=database_binary, database_label=database_label,
            query_ids=torch.as_tensor(np.asarray(query.dataset.indices).copy()),
            database_ids=torch.as_tensor(np.asarray(database.dataset.indices).copy()),
            pr_data=pr_data, per_query_ap=per_query,
        ), destination / "retrieval_outputs.pt")
        report.update(query_evaluated=True, query_metrics=dict(zip(METRICS, map(float, values))),
                      retrieval_outputs_sha256=file_sha256(destination / "retrieval_outputs.pt"))
        (destination / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", help="New directory; default: RUN/final_query")
    parser.add_argument("--data-root", help="Optional relocation of identical raw CIFAR-100 files")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check model/data/provenance without query forwards")
    options = parser.parse_args(argv)
    try:
        report = evaluate(options)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
