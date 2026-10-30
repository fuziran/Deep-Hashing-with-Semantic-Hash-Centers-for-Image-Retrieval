"""Train four matched five-epoch heads on frozen B1+NFD features.

Only training images provide optimizer inputs. Validation/database are encoded
for final symmetric-Hamming scoring; the final query loader is never iterated.
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
from scripts.diagnose_i3_bottleneck import encode
from scripts.diagnose_i3_neighbor_fusion import load_nfd_geometry
from scripts.diagnose_i3_validation import METRICS, checked_run, score_validation
from scripts.diagnose_i3_view_fusion import paired_summary
from utils.experiment import file_sha256, seed_everything
from utils.frozen_b1 import load_frozen_b1
from utils.topology_head import VERSION, MODES, EPOCHS, BATCH_SIZE, fit_head


@torch.inference_mode()
def encode_training(loader, net, device):
    parts, classes, ids = [], [], []
    for batch_index, (weak, labels, image_ids, strong) in enumerate(loader, 1):
        _, first = net.forward_with_features(weak.to(device))
        _, second = net.forward_with_features(strong.to(device))
        if (first.shape != second.shape or first.ndim != 2
                or not torch.isfinite(first).all() or not torch.isfinite(second).all()
                or (first.norm(dim=1) <= 1e-12).any() or (second.norm(dim=1) <= 1e-12).any()):
            raise ValueError("Invalid frozen training features")
        parts.append(torch.stack((F.normalize(first, dim=1), F.normalize(second, dim=1)), dim=1).cpu())
        classes.append(labels.argmax(1).cpu())
        ids.extend(map(int, image_ids.tolist()))
        if batch_index % 25 == 0:
            print(f"Training feature extraction: batch {batch_index}/{len(loader)}", flush=True)
    expected = list(map(int, loader.dataset.indices))
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("Training cache IDs differ from exact training split")
    lookup = {image_id: index for index, image_id in enumerate(ids)}
    order = torch.tensor([lookup[image_id] for image_id in expected])
    return dict(features=torch.cat(parts)[order], classes=torch.cat(classes)[order],
                ids=torch.tensor(expected))


def pack_encoded(result):
    return {key: torch.from_numpy(result[key].copy())
            for key in ("features", "codes", "classes", "ids")}


def validate_cache(cache, identity, datasets, bits):
    if cache.get("identity") != identity:
        raise ValueError("Frozen-feature cache identity differs from this model/data/view policy")
    for name, dataset in datasets.items():
        values = cache[name]
        ids = np.asarray(dataset.indices)
        expected_classes = np.asarray(dataset.labels)[ids]
        features = values["features"]
        if (not np.array_equal(values["ids"].cpu().numpy(), ids)
                or not np.array_equal(values["classes"].cpu().numpy(), expected_classes)
                or len(features) != len(ids) or not torch.isfinite(features).all()
                or features.shape[-1] != 512
                or (features.norm(dim=-1) <= 1e-12).any()):
            raise ValueError(f"Invalid {name} cached IDs, labels or features")
        if name == "training":
            if features.ndim != 3 or features.shape[1] != 2:
                raise ValueError("Training cache must contain exactly two views")
        elif (features.ndim != 2 or values["codes"].shape != (len(ids), bits)
              or not ((values["codes"] == -1) | (values["codes"] == 1)).all()):
            raise ValueError(f"Invalid {name} binary-code cache")


@torch.inference_mode()
def predict_codes(head, features, device, batch_size=1024):
    head.eval()
    parts = []
    for start in range(0, len(features), batch_size):
        logits = head(features[start:start + batch_size].to(device).float())
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite head evaluation logits")
        parts.append(torch.where(logits >= 0, 1, -1).to(torch.int8).cpu().numpy())
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
        raise ValueError("CUDA unavailable")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    deterministic = not config.get("non_deterministic", False)
    seed_everything(config["seed"], deterministic=deterministic)
    args = argparse.Namespace(**config)
    args.variant, args.device, args.num_workers = "b3nvc", device, options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or args.root)).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_topology_head_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del query
        if args.split_hash != config["split_hash"]:
            raise ValueError("Reconstructed split differs from NFD training")
        expected_data = config.get("dataset_content_sha256")
        if expected_data and expected_data != args.dataset_content_sha256:
            raise ValueError("Dataset content differs from NFD training")
        identity = dict(version=VERSION, checkpoint_sha256=run["checkpoint_sha256"],
                        split_hash=args.split_hash, dataset_content_sha256=args.dataset_content_sha256,
                        loader_sha256=file_sha256(ROOT / "data" / "data_loader.py"),
                        collection_seed=config["seed"] + 500003,
                        collection_num_workers=options.num_workers,
                        loader_order_seed=config["seed"],
                        view_policy="existing weak/strong training transforms; two fixed cached views",
                        feature_policy="L2-normalized frozen NFD student features")
        report = dict(version=VERSION, reference_run=str(run["path"]), identity=identity,
                      selected_backbone_epoch=run["summary"]["selected_epoch"],
                      training_images=len(train.dataset), validation_images=len(validation.dataset),
                      database_images=len(database.dataset), epochs_per_head=EPOCHS,
                      modes=list(MODES), code_length=config["code_length"], query_evaluated=False,
                      selection="all trained heads use fixed final epoch; no validation checkpoint selection",
                      protocol="optimizer uses training cache only; validation vs original database; final query never iterated",
                      source_sha256=file_sha256(__file__),
                      head_source_sha256=file_sha256(ROOT / "utils" / "topology_head.py"))
        if options.dry_run:
            report["dry_run"] = True
            return report
        args.num_workers = config["num_workers"]
        _, centers = load_frozen_b1(args)
        if centers.shape == (config["code_length"], config["num_classes"]):
            centers = centers.T
        _, neighbors, prototype_hash = load_nfd_geometry(config, run["path"], torch.device("cpu"))
        report["prototype_sha256"] = prototype_hash
        datasets = dict(training=relation.dataset, validation=validation.dataset, database=database.dataset)
        if options.feature_cache:
            cache_path = Path(options.feature_cache).resolve()
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            net = ResNet(config["code_length"], pretrained=False)
            net.load_state_dict(torch.load(run["checkpoint"], map_location="cpu", weights_only=True), strict=True)
            net.to(device).eval().requires_grad_(False)
            seed_everything(identity["collection_seed"], deterministic=deterministic)
            print("1/3 Extracting two frozen training feature views", flush=True)
            training = encode_training(train, net, device)
            print("2/3 Extracting single-view validation features", flush=True)
            val = pack_encoded(encode(validation, net, device, config["code_length"], description="validation"))
            print("3/3 Extracting single-view database features", flush=True)
            db = pack_encoded(encode(database, net, device, config["code_length"], description="database"))
            del net
            cache = dict(identity=identity, training=training, validation=val, database=db)
            cache_path = output / "frozen_features.pt"
        validate_cache(cache, identity, datasets, config["code_length"])
        val, db = cache["validation"], cache["database"]
        val_classes, db_classes = val["classes"].numpy(), db["classes"].numpy()
        scored = {"original": score_validation(val["codes"].numpy(), db["codes"].numpy(),
                                                 val_classes, db_classes, "original head")}
        for metric, recorded in run["selected"]["validation"].items():
            if metric in scored["original"]["ap"] and abs(
                    scored["original"]["ap"][metric].mean() - recorded) > 1e-5:
                raise ValueError(f"Original {metric} differs from saved validation")
        output.mkdir(parents=True, exist_ok=False)
        if not options.feature_cache:
            torch.save(cache, cache_path)
        report["feature_cache"] = str(cache_path)
        fits = {}

        def progress(mode, record):
            print(f"{mode}: epoch {record['epoch']}/{EPOCHS} " + json.dumps(record), flush=True)

        for mode in MODES:
            print(f"Training {mode}: fixed {EPOCHS} epochs on cached training features", flush=True)
            head, history, fit = fit_head(cache["training"]["features"],
                                          cache["training"]["classes"], centers, neighbors,
                                          mode, config["seed"] + 500021, device, progress=progress)
            val_codes = predict_codes(head, val["features"], device)
            db_codes = predict_codes(head, db["features"], device)
            scored[mode] = score_validation(val_codes, db_codes, val_classes, db_classes, mode)
            torch.save(dict(state_dict={key: value.detach().cpu() for key, value in head.state_dict().items()},
                            fit=fit, identity=identity, feature_dim=512, bits=config["code_length"]),
                       output / f"head_{mode}.pt")
            (output / f"history_{mode}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            fits[mode] = fit
            del head
        report["fits"] = fits
        report["results"] = {
            name: {**{metric: float(values["ap"][metric].mean()) for metric, _ in METRICS},
                   "mean_false_top100": float(values["false_top100"].mean())}
            for name, values in scored.items()
        }
        report["comparison_to_original"] = {
            mode: paired_summary(scored["original"], scored[mode]) for mode in MODES
        }
        report["nfd_vs_controls"] = {
            mode: paired_summary(scored[mode], scored["nfd_topology"])
            for mode in MODES if mode != "nfd_topology"
        }
        report["note"] = (
            "CroVCA-inspired control, not a paper reproduction: NFD features, fixed cached weak/strong "
            "views and symmetric binary Hamming differ from the paper setup. Four heads share "
            "architecture, initialization, batch order and five-epoch budget. Extra topology passes "
            "do not update BN running statistics. Original backbone was selected on this validation "
            "set, so intervals are descriptive. No final query scored, no novelty established.")
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method"] + list(report["results"]["original"]))
            writer.writeheader()
            for name, values in report["results"].items():
                writer.writerow(dict(method=name, **values))
        with (output / "per_query_validation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["validation_id", "class_id"] +
                            [f"{name}_{metric}" for name in scored for metric, _ in METRICS])
            for index, image_id in enumerate(val["ids"].tolist()):
                writer.writerow([image_id, int(val_classes[index])] +
                                [float(values["ap"][metric][index])
                                 for values in scored.values() for metric, _ in METRICS])
        (output / "topology_head_pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache")
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
