"""Screen frozen NFD features for a dense teacher-prototype margin signal.

Only the fixed training and validation splits are encoded. The final query and
database loaders are constructed for split identity but never iterated. This
is a diagnostic, not an I3 training method or a claim of retrieval improvement.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from data.data_loader import load_data, _resolve_cifar100_root
from network import ResNet
from utils.experiment import build_cache_metadata, file_sha256, load_cache, seed_everything
from utils.neighbor_feature_distill import load_nfd
from utils.reliable_relation import tensor_sha256


def margins(features, prototypes, targets):
    if (features.ndim != 2 or prototypes.ndim != 2
            or features.shape[1] != prototypes.shape[1]
            or targets.shape != (len(features),)
            or not torch.isfinite(features).all()
            or not torch.isfinite(prototypes).all()
            or (features.norm(dim=1) <= 1e-12).any()
            or (prototypes.norm(dim=1) <= 1e-12).any()
            or (targets < 0).any() or (targets >= len(prototypes)).any()):
        raise ValueError("Invalid features, prototypes or classes")
    scores = F.normalize(features, dim=1) @ F.normalize(prototypes.to(features), dim=1).T
    own = scores.gather(1, targets[:, None]).squeeze(1)
    other = scores.scatter(1, targets[:, None], -torch.inf)
    nearest = other.argmax(1)
    return own - other.gather(1, nearest[:, None]).squeeze(1), nearest


@torch.inference_mode()
def inspect(loader, student, teacher, prototypes, neighbors, device, description):
    student.eval()
    teacher.eval()
    seen = set()
    count = student_correct = teacher_correct = actionable = neighbor_hard = 0
    student_margin_sum = teacher_margin_sum = 0.
    for batch_index, (images, labels, ids) in enumerate(loader, 1):
        image_ids = list(map(int, ids.tolist()))
        if seen.intersection(image_ids):
            raise ValueError("Duplicate image ID")
        seen.update(image_ids)
        images = images.to(device, non_blocking=True)
        targets = labels.argmax(1).to(device)
        _, student_features = student.forward_with_features(images)
        teacher_features = teacher.forward_features(images)
        sm, nearest = margins(student_features, prototypes, targets)
        tm, _ = margins(teacher_features, prototypes, targets)
        count += len(ids)
        student_correct += int((sm > 0).sum())
        teacher_correct += int((tm > 0).sum())
        actionable += int(((sm < .1) & (tm >= .1)).sum())
        neighbor_hard += int((nearest[:, None] == neighbors[targets]).any(1).sum())
        student_margin_sum += float(sm.sum())
        teacher_margin_sum += float(tm.sum())
        if batch_index % 50 == 0:
            print(f"{description}: batch {batch_index}/{len(loader)}", flush=True)
    expected = set(map(int, loader.dataset.indices))
    if seen != expected or count != len(expected):
        raise ValueError("Diagnostic did not cover exact split image IDs")
    return dict(count=count, student_prototype_accuracy=student_correct / count,
                teacher_prototype_accuracy=teacher_correct / count,
                actionable_fraction=actionable / count,
                nearest_negative_is_nfd_neighbor_fraction=neighbor_hard / count,
                student_mean_margin=student_margin_sum / count,
                teacher_mean_margin=teacher_margin_sum / count)


def analyze(options):
    config_path = Path(options.nfd_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("variant") != "b2nfd" or config.get("dataset") != "cifar-100-new-seg"
            or config.get("seed") != 60 or config.get("code_length") != 32
            or config.get("nfd_alpha", 0) <= 0 or not config.get("paired_stage_seeds")):
        raise ValueError("Expected paired 32-bit seed-60 B1+NFD reference")
    output = Path(options.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = config_path.parent / "best_model_state.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    seed_everything(config["seed"], deterministic=not config.get("non_deterministic", False))
    args = argparse.Namespace(**config)
    args.device = device
    args.num_workers = options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or args.root)).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_i3_proto_screen_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del train, query, database
        if args.split_hash != config["split_hash"]:
            raise ValueError("Split differs from NFD reference")
        if (config.get("dataset_content_sha256")
                and args.dataset_content_sha256 != config["dataset_content_sha256"]):
            raise ValueError("Dataset content differs from NFD reference")
        report = dict(protocol="frozen training and validation features; final query/database never iterated",
                      reference_run=str(config_path.parent),
                      reference_config_sha256=file_sha256(config_path),
                      reference_checkpoint_sha256=file_sha256(checkpoint),
                      split_hash=args.split_hash,
                      dataset_content_sha256=args.dataset_content_sha256,
                      device=str(device), actionable_margin=.1)
        if options.dry_run:
            report["dry_run"] = True
            return report
        nfd = load_nfd(args)
        parent_config = json.loads(Path(config["parent_b1_config"]).read_text(encoding="utf-8"))
        parent_args = argparse.Namespace(**parent_config)
        artifact = load_cache(config["similarity_cache"],
                              build_cache_metadata(parent_args, "similarity"))
        if artifact is None or tensor_sha256(artifact["prototypes"]) != args.nfd_prototypes_sha256:
            raise ValueError("I1 prototype cache identity mismatch")
        prototypes = artifact["prototypes"].to(device)
        student = ResNet(config["code_length"], pretrained=False)
        student.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        student.to(device).eval()
        report["teacher_sha256"] = args.nfd_teacher_sha256
        report["prototype_sha256"] = args.nfd_prototypes_sha256
        report["training"] = inspect(relation, student, nfd.teacher, prototypes,
                                     nfd.neighbors, device, "training")
        report["validation"] = inspect(validation, student, nfd.teacher, prototypes,
                                       nfd.neighbors, device, "validation")
        report["note"] = ("Prototype accuracy is a diagnostic proxy, not retrieval mAP. "
                          "The same validation split selected the frozen checkpoint. "
                          "No model was trained or final query scored.")
        output.mkdir(parents=True, exist_ok=False)
        (output / "prototype_margin.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-config", required=True)
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
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
