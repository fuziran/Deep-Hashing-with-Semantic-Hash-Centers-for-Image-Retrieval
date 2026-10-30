"""Training-only audit of the extra signal in a frozen two-view hash target.

This diagnostic does not train a student. It compares center-crop and fused
center-crop/flip targets for one independently augmented input of each training
image. Validation, query and database loaders are never iterated.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from data.data_loader import load_data, _resolve_cifar100_root, _seed_worker
from network import ResNet
from scripts.diagnose_i3_validation import checked_run
from utils.experiment import file_sha256, seed_everything
from utils.frozen_b1 import load_frozen_b1

VERSION = "i3-training-only-fused-target-signal-v1"


class TrainingFusionViews(Dataset):
    def __init__(self, relation_dataset, strong_transform):
        self.base = relation_dataset
        self.strong_transform = strong_transform
        self.indices = relation_dataset.indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        center, label, image_id = self.base[index]
        strong = self.strong_transform(Image.fromarray(self.base.images[int(image_id)]))
        return center, strong, label, image_id


def target_statistics(single, flipped, strong, target, temperature):
    if (single.ndim != 2 or single.shape != flipped.shape or single.shape != strong.shape
            or target.shape != single.shape or not math.isfinite(temperature)
            or temperature <= 0 or not all(torch.isfinite(x).all()
                                           for x in (single, flipped, strong, target))
            or not ((target == -1) | (target == 1)).all()):
        raise ValueError("Invalid paired logits, center targets or temperature")
    fused = (single + flipped) / 2
    truth = (target + 1) / 2
    single_code, fused_code, strong_code = single >= 0, fused >= 0, strong >= 0
    truth_code = target > 0
    q_single = torch.sigmoid(2 * single / temperature)
    q_fused = torch.sigmoid(2 * fused / temperature)
    p_strong = torch.sigmoid(2 * strong / temperature)

    def kl(teacher):
        t = 2 * teacher / temperature
        s = 2 * strong / temperature
        q = torch.sigmoid(t)
        return (temperature ** 2 * (
            q * (F.logsigmoid(t) - F.logsigmoid(s))
            + (1 - q) * (F.logsigmoid(-t) - F.logsigmoid(-s)))).mean(1)

    bits = single.shape[1]
    # Derivatives with respect to strong-view RAW hash logits, per example.
    # No backbone-gradient or optimization claim is made from these values.
    center_gradient = 2 * (torch.sigmoid(2 * strong) - truth) / bits
    single_gradient = 2 * temperature * (p_strong - q_single) / bits
    fused_gradient = 2 * temperature * (p_strong - q_fused) / bits
    return dict(
        single_wrong_bits=(single_code != truth_code).sum(1),
        fused_wrong_bits=(fused_code != truth_code).sum(1),
        strong_wrong_bits=(strong_code != truth_code).sum(1),
        fusion_changed_bits=(single_code != fused_code).sum(1),
        single_repairs=((single_code == truth_code) & (strong_code != truth_code)).sum(1),
        single_damages=((single_code != truth_code) & (strong_code == truth_code)).sum(1),
        fused_repairs=((fused_code == truth_code) & (strong_code != truth_code)).sum(1),
        fused_damages=((fused_code != truth_code) & (strong_code == truth_code)).sum(1),
        target_probability_gap=(q_fused - q_single).abs().mean(1),
        single_to_strong_kl=kl(single), fused_to_strong_kl=kl(fused),
        center_gradient=center_gradient,
        single_gradient=single_gradient,
        fused_gradient=fused_gradient,
    )


def summarize(parts, bits):
    values = {key: torch.cat([p[key].detach().cpu() for p in parts]) for key in parts[0]}
    count = len(values["single_wrong_bits"])
    result = dict(count=count)
    for name in ("single", "fused", "strong"):
        wrong = values[name + "_wrong_bits"]
        result[name + "_wrong_bit_fraction"] = float(wrong.sum() / (count * bits))
        result[name + "_exact_center_image_fraction"] = float((wrong == 0).float().mean())
    changed = values["fusion_changed_bits"]
    result["fusion_changes_single_image_fraction"] = float((changed > 0).float().mean())
    result["fusion_changes_single_bit_fraction"] = float(changed.sum() / (count * bits))
    for key in ("single_repairs", "single_damages", "fused_repairs", "fused_damages"):
        result[key + "_bit_fraction"] = float(values[key].sum() / (count * bits))
    for key in ("target_probability_gap", "single_to_strong_kl", "fused_to_strong_kl"):
        result[key + "_mean"] = float(values[key].mean())
        result[key + "_p90"] = float(torch.quantile(values[key].float(), .9))
    center = values["center_gradient"].double().flatten()
    single = values["single_gradient"].double().flatten()
    fused = values["fused_gradient"].double().flatten()
    center_norm = float(center.norm())
    result["logit_gradient_diagnostic"] = dict(
        center_norm=center_norm,
        single_target_norm=float(single.norm()), fused_target_norm=float(fused.norm()),
        fusion_extra_norm=float((fused - single).norm()),
        fusion_extra_over_center=(float((fused - single).norm()) / center_norm
                                  if center_norm > 1e-12 else None),
        fused_vs_center_cosine=(float(F.cosine_similarity(fused[None], center[None]))
                                if center_norm > 1e-12 and fused.norm() > 1e-12 else None),
        scope="raw hash logits only; center BCE excludes quantization and NFD; not parameter gradients")
    return result, values


@torch.inference_mode()
def inspect(loader, student, centers, device, bits, temperature):
    parts, image_ids, classes = [], [], []
    student.eval().requires_grad_(False)
    for batch_index, (center, strong, labels, ids) in enumerate(loader, 1):
        center, strong = center.to(device), strong.to(device)
        targets = labels.argmax(1).to(device)
        single = student(center)
        flipped = student(center.flip(-1))
        strong_logits = student(strong)
        if single.shape != (len(ids), bits):
            raise ValueError("Invalid student hash-logit shape")
        batch = target_statistics(single, flipped, strong_logits,
                                  centers[targets].to(single), temperature)
        parts.append({key: value.cpu() for key, value in batch.items()})
        image_ids.extend(map(int, ids.tolist()))
        classes.extend(map(int, targets.cpu().tolist()))
        if batch_index % 25 == 0:
            print(f"Training target signal: batch {batch_index}/{len(loader)}", flush=True)
    if image_ids != list(map(int, loader.dataset.indices)):
        raise ValueError("Diagnostic did not cover exact ordered training IDs")
    summary, values = summarize(parts, bits)
    return summary, values, image_ids, classes


def analyze(options):
    run = checked_run(options.nfd_run)
    config = run["config"]
    if (config["variant"] != "b2nfd" or config["code_length"] != 32
            or config["seed"] != 60 or config.get("nfd_alpha", 0) <= 0):
        raise ValueError("Expected paired 32-bit seed-60 B1+NFD reference")
    output = Path(options.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    device = torch.device("cpu" if options.cpu else f"cuda:{options.gpu}")
    seed_everything(config["seed"], deterministic=not config.get("non_deterministic", False))
    args = argparse.Namespace(**config)
    args.variant = "b3nvc"  # Reuse the existing, fixed strong transform only.
    args.device, args.num_workers = device, options.num_workers
    args.root = str(Path(_resolve_cifar100_root(options.data_root or args.root)).resolve())
    with tempfile.TemporaryDirectory(prefix="shc_fused_target_signal_") as scratch:
        args.output_dir = scratch
        train, relation, validation, query, database, *_ = load_data(args)
        del validation, query, database
        if args.split_hash != config["split_hash"]:
            raise ValueError("Split differs from NFD training")
        if (config.get("dataset_content_sha256")
                and args.dataset_content_sha256 != config["dataset_content_sha256"]):
            raise ValueError("Dataset content differs from NFD training")
        report = dict(
            version=VERSION, reference_run=str(run["path"]),
            checkpoint_sha256=run["checkpoint_sha256"],
            diagnostic_source_sha256=file_sha256(__file__),
            split_hash=args.split_hash, dataset_content_sha256=args.dataset_content_sha256,
            selected_epoch=run["summary"]["selected_epoch"],
            training_count=len(relation.dataset), temperature=options.temperature,
            protocol="training subset only; validation/query/database never iterated",
            view_policy="center crop, its horizontal flip, existing independent strong augmentation",
            device=str(device))
        if options.dry_run:
            report["dry_run"] = True
            return report
        dataset = TrainingFusionViews(relation.dataset, train.dataset.strong_transform)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=options.num_workers,
                            worker_init_fn=_seed_worker,
                            generator=torch.Generator().manual_seed(config["seed"] + 300007),
                            pin_memory=device.type == "cuda")
        args.num_workers = config["num_workers"]
        _, centers = load_frozen_b1(args)
        if centers.shape == (config["code_length"], config["num_classes"]):
            centers = centers.T
        if centers.shape != (config["num_classes"], config["code_length"]):
            raise ValueError("Invalid frozen B1 centers")
        student = ResNet(config["code_length"], pretrained=False)
        student.load_state_dict(torch.load(run["checkpoint"], map_location="cpu",
                                           weights_only=True), strict=True)
        student.to(device).eval().requires_grad_(False)
        training, values, image_ids, classes = inspect(
            loader, student, centers.to(device), device,
            config["code_length"], options.temperature)
        report["training"] = training
        report["note"] = (
            "Signal audit at an already fitted checkpoint, not a prediction of from-scratch "
            "training or retrieval efficacy. Output-logit gradients are unweighted diagnostics, "
            "not full-network gradients. No optimizer, model update, retrieval evaluation or "
            "novelty claim. Cross-view hash distillation has prior art.")
        output.mkdir(parents=True, exist_ok=False)
        keys = [key for key in values if not key.endswith("gradient")]
        with (output / "per_image_training.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["training_id", "class_id"] + keys)
            for index, image_id in enumerate(image_ids):
                writer.writerow([image_id, classes[index]] + [float(values[key][index]) for key in keys])
        (output / "fused_target_signal.json").write_text(
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
    parser.add_argument("--temperature", type=float, default=2.)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if (options.num_workers < 0 or not math.isfinite(options.temperature)
            or options.temperature <= 0):
        parser.error("Invalid workers or temperature")
    try:
        report = analyze(options)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
