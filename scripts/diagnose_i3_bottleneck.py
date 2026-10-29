"""Audit a frozen hash student's representation and code bottlenecks.

This is exploratory: only validation images and the original database gallery
are encoded. The final query loader is constructed for split verification but
never iterated. No teacher, optimizer, or training step is constructed.
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
from network import ResNet
from scripts.diagnose_i3_validation import METRICS, checked_run, score_validation
from utils.experiment import file_sha256, seed_everything


def encode(loader, net, device, bits, flip=False, description=""):
    features, logits, codes, classes, ids = [], [], [], [], []
    flipped = []
    net.eval()
    with torch.inference_mode():
        for batch_index, (images, labels, indices) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            output, embedding = net.forward_with_features(images)
            if (output.ndim != 2 or output.shape != (len(images), bits)
                    or embedding.ndim != 2 or len(embedding) != len(images)
                    or not torch.isfinite(output).all()
                    or not torch.isfinite(embedding).all()
                    or (embedding.norm(dim=1) <= 1e-12).any()
                    or (output.norm(dim=1) <= 1e-12).any()):
                raise ValueError("Invalid frozen student features or hash logits")
            features.append(F.normalize(embedding.float(), dim=1).cpu().numpy())
            logits.append(F.normalize(output.float(), dim=1).cpu().numpy())
            binary = torch.sign(output)
            binary[binary == 0] = 1
            codes.append(binary.to(torch.int8).cpu().numpy())
            classes.append(labels.argmax(1).numpy())
            ids.append(indices.numpy())
            if flip:
                flipped_output, _ = net.forward_with_features(torch.flip(images, dims=(-1,)))
                if flipped_output.shape != output.shape or not torch.isfinite(flipped_output).all():
                    raise ValueError("Invalid horizontally flipped student logits")
                flipped_binary = torch.sign(flipped_output)
                flipped_binary[flipped_binary == 0] = 1
                flipped.append(flipped_binary.to(torch.int8).cpu().numpy())
            if description and batch_index % 100 == 0:
                print(f"Encoding {description}: batch {batch_index}/{len(loader)}", flush=True)
    result = dict(features=np.concatenate(features), logits=np.concatenate(logits),
                  codes=np.concatenate(codes), classes=np.concatenate(classes),
                  ids=np.concatenate(ids))
    expected_ids = np.asarray(loader.dataset.indices)
    expected_classes = np.asarray(loader.dataset.labels)[expected_ids]
    if (not np.array_equal(result["ids"], expected_ids)
            or not np.array_equal(result["classes"], expected_classes)):
        raise ValueError("Encoded IDs/classes differ from the audited dataset")
    if flip:
        result["flipped_codes"] = np.concatenate(flipped)
    return result


def score_cosine(query, gallery, query_classes, gallery_classes, device,
                 description="", chunk_size=16):
    """Use the same AP definition and stable rank order as the Hamming evaluator."""
    if (query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]
            or len(query) != len(query_classes) or len(gallery) != len(gallery_classes)
            or not np.isfinite(query).all() or not np.isfinite(gallery).all()
            or (np.linalg.norm(query, axis=1) <= 1e-12).any()
            or (np.linalg.norm(gallery, axis=1) <= 1e-12).any()):
        raise ValueError("Invalid cosine retrieval arrays")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    ap = {name: np.empty(len(query), dtype=np.float64) for name, _ in METRICS}
    false_top100 = np.empty(len(query), dtype=np.int32)
    gallery_t = torch.from_numpy(gallery.T.copy()).to(device)
    k100 = min(100, len(gallery))
    for start in range(0, len(query), chunk_size):
        batch = torch.from_numpy(query[start:start + chunk_size].copy()).to(device)
        scores = (batch @ gallery_t).cpu().numpy()
        for offset, row in enumerate(scores):
            i = start + offset
            order = np.argsort(-row, kind="stable")
            relevant = gallery_classes[order] == query_classes[i]
            ranks = np.flatnonzero(relevant) + 1
            if not len(ranks):
                raise ValueError(f"Validation image {i} has no relevant gallery image")
            for name, cutoff in METRICS:
                selected = ranks if cutoff == -1 else ranks[ranks <= cutoff]
                ap[name][i] = (np.arange(1, len(selected) + 1) / selected).mean() if len(selected) else 0.
            false_top100[i] = k100 - int(relevant[:k100].sum())
        if description and (start // chunk_size + 1) % 10 == 0:
            print(f"Scoring {description}: validation {min(start + chunk_size, len(query))}/{len(query)}", flush=True)
    return dict(ap=ap, false_top100=false_top100)


def analyze(options):
    run = checked_run(options.run)
    config = run["config"]
    output = Path(options.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --cpu for a small diagnostic")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    seed_everything(config["seed"], deterministic=not config.get("non_deterministic", False))
    args = argparse.Namespace(**config)
    args.device = device
    args.num_workers = options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or args.root)).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_i3_bottleneck_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del train, query
        if args.split_hash != config["split_hash"]:
            raise ValueError("Reconstructed split differs from saved run")
        expected_data = config.get("dataset_content_sha256")
        if expected_data and expected_data != args.dataset_content_sha256:
            raise ValueError("Dataset pixels/labels differ from saved run")
        report = dict(
            protocol="validation vs original database; final query never iterated",
            run_dir=str(run["path"]), variant=config["variant"],
            selected_epoch=run["summary"]["selected_epoch"],
            checkpoint_sha256=run["checkpoint_sha256"],
            training_source_sha256=config.get("source_sha256"),
            diagnostic_source_sha256=file_sha256(__file__),
            split_hash=args.split_hash, dataset_content_sha256=args.dataset_content_sha256,
            code_length=config["code_length"], seed=config["seed"],
            training_count=len(relation.dataset), validation_count=len(validation.dataset),
            database_count=len(database.dataset),
            selected_validation=run["selected"]["validation"],
            selection_receipt_present=run["selection_receipt_present"],
            device=str(device),
        )
        if options.dry_run:
            report["dry_run"] = True
            return report
        net = ResNet(config["code_length"], pretrained=False)
        state = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
        if (not isinstance(state, dict) or not state
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                       for value in state.values())):
            raise ValueError("Checkpoint is not a finite student state_dict")
        net.load_state_dict(state, strict=True)
        net.to(device).eval()
        val = encode(validation, net, device, config["code_length"], flip=True,
                     description="validation and horizontal flip")
        training = encode(relation, net, device, config["code_length"], flip=True,
                          description="training subset and horizontal flip")
        db = encode(database, net, device, config["code_length"], description="database")
        del net
        scored = {
            "feature_cosine": score_cosine(val["features"], db["features"],
                                            val["classes"], db["classes"], device,
                                            "512d feature cosine", options.chunk_size),
            "continuous_code_cosine": score_cosine(val["logits"], db["logits"],
                                                    val["classes"], db["classes"], device,
                                                    "continuous code cosine", options.chunk_size),
            "binary_hamming": score_validation(val["codes"], db["codes"],
                                                val["classes"], db["classes"], "binary Hamming"),
        }
        for metric, recorded in run["selected"]["validation"].items():
            if metric in scored["binary_hamming"]["ap"] and abs(
                    float(scored["binary_hamming"]["ap"][metric].mean()) - recorded) > 1e-5:
                raise ValueError(f"Recomputed Hamming {metric} differs from saved validation")
        changed = val["codes"] != val["flipped_codes"]
        training_changed = training["codes"] != training["flipped_codes"]
        flip_fraction = changed.mean(axis=1)
        report["retrieval"] = {
            name: {**{metric: float(result["ap"][metric].mean()) for metric, _ in METRICS},
                   "mean_false_top100": float(result["false_top100"].mean())}
            for name, result in scored.items()
        }
        report["horizontal_flip"] = dict(
            validation_mean_changed_bit_fraction=float(changed.mean()),
            validation_fraction_images_with_any_changed_bit=float(changed.any(axis=1).mean()),
            validation_per_bit_changed_fraction=[float(x) for x in changed.mean(axis=0)],
            training_mean_changed_bit_fraction=float(training_changed.mean()),
            training_fraction_images_with_any_changed_bit=float(training_changed.any(axis=1).mean()),
            training_per_bit_changed_fraction=[float(x) for x in training_changed.mean(axis=0)],
        )
        report["note"] = ("Exploratory frozen-model analysis; feature/continuous cosine and binary Hamming "
                          "are different retrieval rules, so their score gaps are diagnostic rather than "
                          "a causal decomposition. Center-crop horizontal flip is one view perturbation only. "
                          "The validation set selected this checkpoint; no final query is evaluated.")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "per_query_validation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["validation_id", "class_id", "flip_changed_bit_fraction"]
                            + [f"{name}_{metric}" for name in scored for metric, _ in METRICS]
                            + [f"{name}_false_top100" for name in scored])
            for i, image_id in enumerate(val["ids"]):
                writer.writerow([int(image_id), int(val["classes"][i]), float(flip_fraction[i])]
                                + [float(result["ap"][metric][i]) for result in scored.values()
                                   for metric, _ in METRICS]
                                + [int(result["false_top100"][i]) for result in scored.values()])
        (output / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Frozen ordinary-FD or NFD run directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if options.num_workers < 0 or options.chunk_size < 1:
        parser.error("--num-workers must be nonnegative and --chunk-size must be positive")
    try:
        report = analyze(options)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
