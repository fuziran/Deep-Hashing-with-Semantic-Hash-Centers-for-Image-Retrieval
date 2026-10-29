"""Compare two frozen CIFAR-100 hash students on validation vs database only.

Historical runs can have different training-source hashes. This diagnostic
checks each run's saved selection, checkpoint and data split, and never iterates
the final query loader or trains a model. Its statistics are exploratory.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.data_loader import load_data, _resolve_cifar100_root
from network import ResNet
from utils.experiment import file_sha256, seed_everything, selection_receipt

METRICS = (("mAP@ALL", -1), ("mAP@100", 100), ("mAP@1000", 1000))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_run(path):
    path = Path(path).resolve()
    config = read_json(path / "config.json")
    summary = read_json(path / "metrics.json")
    history = read_json(path / "training_history.json")
    if (config.get("variant") not in ("b2nfd", "b3fcd")
            or config.get("dataset") != "cifar-100-new-seg"
            or config.get("num_classes") != 100
            or config.get("topK") != [-1, 100, 1000]
            or not config.get("paired_stage_seeds")):
        raise ValueError(f"Unexpected paired CIFAR-100 run: {path}")
    if (not isinstance(history, list) or len(history) != config["epoch"]
            or [r.get("epoch") for r in history] != list(range(1, config["epoch"] + 1))):
        raise ValueError(f"Incomplete training history: {path}")
    validated = [r for r in history if "validation" in r]
    if not validated or any(
            not math.isfinite(r["validation"][name])
            for r in validated for name, _ in METRICS):
        raise ValueError(f"Invalid validation history: {path}")
    best = max(validated, key=lambda r: r["validation"]["mAP@ALL"])
    if (summary.get("selection_metric") != "validation mAP@ALL"
            or summary.get("selected_epoch") != best["epoch"]
            or summary.get("best_validation_mAP@ALL") != best["validation"]["mAP@ALL"]):
        raise ValueError(f"Selected model does not match validation history: {path}")
    receipt = selection_receipt(path)
    has_receipt = summary.get("selection_receipt_version") == 1
    if has_receipt and any(summary.get(key) != value or value is None
                           for key, value in receipt.items()):
        raise ValueError(f"Selection receipt mismatch: {path}")
    checkpoint = path / "best_model_state.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return dict(path=path, config=config, summary=summary, selected=best,
                checkpoint=checkpoint, checkpoint_sha256=file_sha256(checkpoint),
                selection_receipt_present=has_receipt)


def encode(loader, net, device, bits, description=""):
    codes, logits, classes, ids = [], [], [], []
    net.eval()
    with torch.inference_mode():
        for batch_index, (images, labels, indices) in enumerate(loader, 1):
            output = net(images.to(device))
            if output.ndim != 2 or output.shape[1] != bits or not torch.isfinite(output).all():
                raise ValueError("Invalid model hash logits")
            output = output.cpu()
            binary = torch.sign(output)
            binary[binary == 0] = 1
            codes.append(binary.to(torch.int8).numpy())
            logits.append(output.numpy())
            classes.append(labels.argmax(1).numpy())
            ids.append(indices.numpy())
            if description and batch_index % 100 == 0:
                print(f"Encoding {description}: batch {batch_index}/{len(loader)}", flush=True)
    result = tuple(np.concatenate(parts) for parts in (codes, logits, classes, ids))
    _, _, actual_classes, actual_ids = result
    expected_ids = np.asarray(loader.dataset.indices)
    expected_classes = np.asarray(loader.dataset.labels)[expected_ids]
    if (not np.array_equal(actual_ids, expected_ids)
            or not np.array_equal(actual_classes, expected_classes)):
        raise ValueError("Encoded IDs or labels differ from the audited loader")
    return result


def score_validation(query_codes, database_codes, query_classes, database_classes,
                     description=""):
    """Match stable Hamming AP while sorting each validation query just once."""
    if (query_codes.ndim != 2 or database_codes.ndim != 2
            or query_codes.shape[1] != database_codes.shape[1]
            or len(query_codes) != len(query_classes)
            or len(database_codes) != len(database_classes)
            or not np.isin(query_codes, (-1, 1)).all()
            or not np.isin(database_codes, (-1, 1)).all()):
        raise ValueError("Invalid binary codes or labels")
    bits = query_codes.shape[1]
    database_t = database_codes.astype(np.int16).T
    ap = {name: np.empty(len(query_codes), dtype=np.float64) for name, _ in METRICS}
    false_top100 = np.empty(len(query_codes), dtype=np.int32)
    boundary_ties = np.empty(len(query_codes), dtype=np.int32)
    boundary_extends = np.empty(len(query_codes), dtype=bool)
    k100 = min(100, len(database_codes))
    for i in range(len(query_codes)):
        distance = (bits - query_codes[i].astype(np.int16) @ database_t) // 2
        order = np.argsort(distance, kind="stable")
        relevant = database_classes[order] == query_classes[i]
        ranks = np.flatnonzero(relevant) + 1
        if len(ranks) == 0:
            raise ValueError(f"Validation query {i} has no relevant database image")
        for name, cutoff in METRICS:
            selected = ranks if cutoff == -1 else ranks[ranks <= cutoff]
            ap[name][i] = (np.arange(1, len(selected) + 1) / selected).mean() if len(selected) else 0.
        false_top100[i] = k100 - int(relevant[:k100].sum())
        boundary = distance[order[k100 - 1]]
        boundary_ties[i] = int((distance == boundary).sum())
        boundary_extends[i] = bool((distance[order[k100:]] == boundary).any())
        if description and (i + 1) % 100 == 0:
            print(f"Scoring {description}: validation {i + 1}/{len(query_codes)}", flush=True)
    return dict(ap=ap, false_top100=false_top100, boundary_ties=boundary_ties,
                boundary_extends=boundary_extends)


def analyze(options):
    baseline, candidate = checked_run(options.baseline_run), checked_run(options.candidate_run)
    a, b = baseline["config"], candidate["config"]
    for key in ("dataset", "seed", "code_length", "num_classes", "val_per_class",
                "resize_size", "crop_size", "split_hash", "topK"):
        if a.get(key) != b.get(key):
            raise ValueError(f"Runs have different {key}")
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --cpu for a diagnostic run")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    seed_everything(a["seed"], deterministic=not a.get("non_deterministic", False))
    args = argparse.Namespace(**a)
    args.device = device
    args.num_workers = options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or a["root"])).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_validation_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        # Query is constructed for split verification, but is never iterated.
        del train, relation, query
        if args.split_hash != a["split_hash"]:
            raise ValueError("Reconstructed split differs from training")
        for run in (baseline, candidate):
            expected = run["config"].get("dataset_content_sha256")
            if expected and expected != args.dataset_content_sha256:
                raise ValueError(f"Dataset content differs from {run['path']}")
        report = dict(
            protocol="validation vs original database; final query never iterated",
            split_hash=args.split_hash, dataset_content_sha256=args.dataset_content_sha256,
            validation_count=len(validation.dataset), database_count=len(database.dataset),
            code_length=a["code_length"], seed=a["seed"],
            methods={}, paired_bootstrap_95={},
        )
        for name, run in (("baseline", baseline), ("candidate", candidate)):
            report["methods"][name] = dict(
                run_dir=str(run["path"]), variant=run["config"]["variant"],
                selected_epoch=run["summary"]["selected_epoch"],
                selected_validation=run["selected"]["validation"],
                checkpoint_sha256=run["checkpoint_sha256"],
                training_source_sha256=run["config"].get("source_sha256"),
                selection_receipt_present=run["selection_receipt_present"],
            )
        if options.dry_run:
            report["dry_run"] = True
            return report
        outputs = {}
        for name, run in (("baseline", baseline), ("candidate", candidate)):
            net = ResNet(a["code_length"], pretrained=False)
            state = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)
            net.to(device).eval()
            v_codes, v_logits, v_classes, v_ids = encode(
                validation, net, device, a["code_length"], f"{name} validation")
            d_codes, _, d_classes, d_ids = encode(
                database, net, device, a["code_length"], f"{name} database")
            del net
            scored = score_validation(v_codes, d_codes, v_classes, d_classes, name)
            means = {key: float(value.mean()) for key, value in scored["ap"].items()}
            for key, recorded in run["selected"]["validation"].items():
                if key in means and abs(means[key] - recorded) > 1e-5:
                    raise ValueError(f"Recomputed {name} {key} differs from saved validation")
            report["methods"][name].update(
                recomputed_validation=means,
                mean_false_top100=float(scored["false_top100"].mean()),
                mean_boundary_tie_size=float(scored["boundary_ties"].mean()),
                fraction_boundary_tie_extends=float(scored["boundary_extends"].mean()),
                mean_absolute_logit=float(np.abs(v_logits).mean()),
                fraction_logits_abs_lt_0_1=float((np.abs(v_logits) < .1).mean()),
            )
            outputs[name] = dict(scored=scored, classes=v_classes, ids=v_ids,
                                 database_ids=d_ids, database_classes=d_classes)
        if (not np.array_equal(outputs["baseline"]["ids"], outputs["candidate"]["ids"])
                or not np.array_equal(outputs["baseline"]["classes"], outputs["candidate"]["classes"])
                or not np.array_equal(outputs["baseline"]["database_ids"], outputs["candidate"]["database_ids"])):
            raise ValueError("Runs did not encode identical validation/database IDs")
        rng = np.random.default_rng(60)
        for metric, _ in METRICS:
            diff = outputs["candidate"]["scored"]["ap"][metric] - outputs["baseline"]["scored"]["ap"][metric]
            bootstrap = np.array([diff[rng.integers(len(diff), size=len(diff))].mean()
                                  for _ in range(2000)])
            report["paired_bootstrap_95"][metric] = dict(
                delta=float(diff.mean()), interval=list(map(float, np.quantile(bootstrap, [.025, .975]))))
        options.output_dir.mkdir(parents=True, exist_ok=False)
        with (options.output_dir / "per_query_validation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["validation_id", "class_id"] +
                            [f"{name}_{metric}" for name in ("baseline", "candidate")
                             for metric, _ in METRICS] +
                            ["baseline_false_top100", "candidate_false_top100",
                             "baseline_boundary_ties", "candidate_boundary_ties"])
            for i, image_id in enumerate(outputs["baseline"]["ids"]):
                writer.writerow([int(image_id), int(outputs["baseline"]["classes"][i])] +
                                [float(outputs[name]["scored"]["ap"][metric][i])
                                 for name in ("baseline", "candidate") for metric, _ in METRICS] +
                                [int(outputs[name]["scored"][key][i])
                                 for key in ("false_top100", "boundary_ties")
                                 for name in ("baseline", "candidate")])
        classes = outputs["baseline"]["classes"]
        class_rows = []
        for class_id in sorted(set(map(int, classes))):
            mask = classes == class_id
            row = dict(class_id=class_id, n_queries=int(mask.sum()))
            for metric, _ in METRICS:
                values = [float(outputs[name]["scored"]["ap"][metric][mask].mean())
                          for name in ("baseline", "candidate")]
                row["baseline_" + metric], row["candidate_" + metric] = values
                row["delta_" + metric] = values[1] - values[0]
            class_rows.append(row)
        with (options.output_dir / "per_class_validation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
            writer.writeheader()
            writer.writerows(class_rows)
        report["class_delta_all_top10"] = sorted(class_rows, key=lambda r: -r["delta_mAP@ALL"])[:10]
        report["class_delta_all_bottom10"] = sorted(class_rows, key=lambda r: r["delta_mAP@ALL"])[:10]
        report["note"] = "Exploratory: the same validation set selected both checkpoints; bootstrap is descriptive, not independent confirmation."
        (options.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    return report


if __name__ == "__main__":
    main()
