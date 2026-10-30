"""Frozen validation screen for NFD-neighbor confidence versus equal view fusion.

Both views use the same selected NFD student. The final query loader is never
iterated. No training or model selection is performed by this script.
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
from scripts.diagnose_i3_view_fusion import binarize, paired_summary
from utils.experiment import build_cache_metadata, file_sha256, load_cache, seed_everything
from utils.neighbor_view_fusion import VERSION, fused_logits
from utils.reliable_relation import tensor_sha256


def load_nfd_geometry(config, run_path, device):
    parent = json.loads(Path(config["parent_b1_config"]).read_text(encoding="utf-8"))
    parent_args = argparse.Namespace(**parent)
    artifact = load_cache(config["similarity_cache"],
                          build_cache_metadata(parent_args, "similarity"))
    if artifact is None or not isinstance(artifact, dict):
        raise ValueError("Missing audited I1 relation artifact")
    prototypes = artifact["prototypes"]
    prototype_hash = tensor_sha256(prototypes)
    if prototype_hash != config["nfd_prototypes_sha256"]:
        raise ValueError("I1 prototypes differ from NFD training")
    projection_path = Path(run_path) / "nfd_projection.pt"
    if not projection_path.is_file():
        raise FileNotFoundError("NFD projection artifact missing from run directory")
    projection = torch.load(projection_path, map_location="cpu", weights_only=True)
    identity = projection["identity"]
    if (identity["prototypes_sha256"] != prototype_hash
            or identity["basis_sha256"] != config["nfd_basis_sha256"]
            or identity["similarity_sha256"] != config["similarity_content_sha256"]):
        raise ValueError("NFD geometry provenance mismatch")
    neighbors = projection["neighbors"]
    if (neighbors.shape != (config["num_classes"], config["nfd_topk"])
            or neighbors.dtype != torch.long):
        raise ValueError("Invalid frozen NFD neighbor matrix")
    return prototypes.to(device), neighbors.to(device), prototype_hash


@torch.inference_mode()
def encode_policies(loader, student, prototypes, neighbors, device, bits, description):
    names = ("original", "uniform", "global_prototype", "nfd_neighbor")
    arrays = {name: [] for name in names}
    classes, ids, global_weights, neighbor_weights = [], [], [], []
    student.eval()
    for batch_index, (images, labels, image_ids) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        logits, features = student.forward_with_features(images)
        flipped_logits, flipped_features = student.forward_with_features(images.flip(-1))
        if (logits.shape != (len(images), bits) or flipped_logits.shape != logits.shape
                or features.shape != (len(images), prototypes.shape[1])
                or flipped_features.shape != features.shape):
            raise ValueError("Invalid student view output shapes")
        fused = fused_logits(logits, flipped_logits, features, flipped_features,
                             prototypes, neighbors)
        arrays["original"].append(binarize(logits.cpu().numpy()))
        for name in names[1:]:
            arrays[name].append(binarize(fused[name].cpu().numpy()))
        classes.append(labels.argmax(1).numpy())
        ids.append(image_ids.numpy())
        global_weights.append(fused["global_weight"].cpu().numpy())
        neighbor_weights.append(fused["neighbor_weight"].cpu().numpy())
        if description and batch_index % 100 == 0:
            print(f"Encoding {description}: batch {batch_index}/{len(loader)}", flush=True)
    result = {name: np.concatenate(parts) for name, parts in arrays.items()}
    result["classes"] = np.concatenate(classes)
    result["ids"] = np.concatenate(ids)
    result["global_weight"] = np.concatenate(global_weights)
    result["neighbor_weight"] = np.concatenate(neighbor_weights)
    expected_ids = np.asarray(loader.dataset.indices)
    expected_classes = np.asarray(loader.dataset.labels)[expected_ids]
    if (not np.array_equal(result["ids"], expected_ids)
            or not np.array_equal(result["classes"], expected_classes)):
        raise ValueError("Encoded IDs/classes differ from audited split")
    return result


def analyze(options):
    run = checked_run(options.nfd_run)
    config = run["config"]
    if (config["variant"] != "b2nfd" or config["dataset"] != "cifar-100-new-seg"
            or config["code_length"] != 32 or config["seed"] != 60):
        raise ValueError("Expected paired 32-bit seed-60 B1+NFD run")
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
    with tempfile.TemporaryDirectory(prefix="shc_i3_neighbor_fusion_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del train, relation, query
        if args.split_hash != config["split_hash"]:
            raise ValueError("Reconstructed split differs from NFD training")
        expected_data = config.get("dataset_content_sha256")
        if expected_data and args.dataset_content_sha256 != expected_data:
            raise ValueError("Dataset content differs from NFD training")
        report = dict(version=VERSION, run_dir=str(run["path"]),
                      selected_epoch=run["summary"]["selected_epoch"],
                      checkpoint_sha256=run["checkpoint_sha256"],
                      diagnostic_source_sha256=file_sha256(__file__),
                      split_hash=args.split_hash,
                      dataset_content_sha256=args.dataset_content_sha256,
                      validation_count=len(validation.dataset),
                      database_count=len(database.dataset),
                      selected_validation=run["selected"]["validation"],
                      device=str(device),
                      protocol="validation vs original database; final query never iterated",
                      policy="same two student views, 32-bit Hamming; per-view weight in [0.25,0.75]")
        if options.dry_run:
            report["dry_run"] = True
            return report
        prototypes, neighbors, prototype_hash = load_nfd_geometry(
            config, run["path"], device)
        report["prototype_sha256"] = prototype_hash
        report["nfd_basis_sha256"] = config["nfd_basis_sha256"]
        student = ResNet(config["code_length"], pretrained=False)
        student.load_state_dict(torch.load(run["checkpoint"], map_location="cpu",
                                           weights_only=True), strict=True)
        student.to(device).eval()
        val = encode_policies(validation, student, prototypes, neighbors, device,
                              config["code_length"], "validation")
        db = encode_policies(database, student, prototypes, neighbors, device,
                             config["code_length"], "database")
        del student
        names = ("original", "uniform", "global_prototype", "nfd_neighbor")
        scored = {name: score_validation(val[name], db[name], val["classes"],
                                         db["classes"], name) for name in names}
        for metric, recorded in run["selected"]["validation"].items():
            if metric in scored["original"]["ap"] and abs(
                    scored["original"]["ap"][metric].mean() - recorded) > 1e-5:
                raise ValueError(f"Original {metric} differs from saved validation")
        report["comparison_to_original"] = {
            name: paired_summary(scored["original"], scored[name])
            for name in names[1:]
        }
        report["comparison_to_uniform"] = {
            name: paired_summary(scored["uniform"], scored[name])
            for name in ("global_prototype", "nfd_neighbor")
        }
        report["changed_vs_uniform"] = {
            name: dict(validation=float((val[name] != val["uniform"]).any(1).mean()),
                       database=float((db[name] != db["uniform"]).any(1).mean()))
            for name in ("global_prototype", "nfd_neighbor")
        }
        report["view_weights"] = {
            name: dict(validation_mean=float(val[key].mean()),
                       database_mean=float(db[key].mean()),
                       validation_mean_abs_from_half=float(np.abs(val[key] - .5).mean()))
            for name, key in (("global_prototype", "global_weight"),
                              ("nfd_neighbor", "neighbor_weight"))
        }
        report["mean_false_top100"] = {
            name: float(scored[name]["false_top100"].mean()) for name in names
        }
        report["note"] = (
            "Exploratory on checkpoint-selection validation; paired intervals are descriptive. "
            "Simple equal fusion is the cost-matched control. Prototype confidence and "
            "adaptive view fusion have prior art; this screen alone establishes no novelty. "
            "No model was trained, final query was not scored.")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "per_query_validation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["validation_id", "class_id"] +
                            [f"{name}_{metric}" for name in names for metric, _ in METRICS])
            for i, image_id in enumerate(val["ids"]):
                writer.writerow([int(image_id), int(val["classes"][i])] +
                                [float(scored[name]["ap"][metric][i])
                                 for name in names for metric, _ in METRICS])
        (output / "neighbor_view_fusion.json").write_text(
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
