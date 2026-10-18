import argparse
import math
import os

# Must be set before the first CUDA BLAS workspace is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from GenerateSemanticHashCenters import GenerateSemanticHashCenters
from GenerateSimilarityMatrix import GenerateSimilarityMatrix
from data.data_loader import load_data
from network import ResNet
from train import train_val
from utils.experiment import (
    current_git_commit,
    seed_everything,
    stable_config_hash,
    source_sha256,
    write_run_manifest,
)


def load_config():
    parser = argparse.ArgumentParser(description="Audited SHC B0/B1 and NABirds-B I2-NFD experiment")
    parser.add_argument("--seed", default=60, type=int)
    parser.add_argument("--variant", choices=("b0", "b1", "b2nfd"), default="b0")
    parser.add_argument("--relation-mode",
                        choices=("full", "two_view", "weighted", "fixed_fusion", "feature_only"),
                        default="full")
    parser.add_argument("--relation-kappa", type=float, default=30.0)
    parser.add_argument("--relation-temperature", type=float, default=1.0)
    parser.add_argument("--relation-weight-floor", type=float, default=0.05)
    parser.add_argument("--paired-stage-seeds", action="store_true",
                        help="Reset hash-stage RNG and training order identically for B0/B1.")
    parser.add_argument("--info", default="[SHC-B0]", type=str)
    parser.add_argument("--dataset", default="cifar-100-new-seg", type=str)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--root", default="./data/cifar-100-python", type=str)
    parser.add_argument("--code-length", default=32, type=int)
    parser.add_argument("--lr", default=7e-5, type=float)
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--classify-epoch", "--classify_epoch", default=300, type=int)
    parser.add_argument("--test-map", "--test_map", default=5, type=int)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--lambd", default=1e-4, type=float)
    parser.add_argument("--topK", nargs="+", default=[-1, 100, 1000], type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--resize-size", default=256, type=int)
    parser.add_argument("--crop-size", default=224, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--val-per-class", default=None, type=int,
        help="Training images held out per class: CIFAR-100 defaults to 10, NABirds-B to 2.",
    )
    parser.add_argument(
        "--mask-strategy",
        choices=("predicted_argmax", "ground_truth"),
        default="predicted_argmax",
        help="Paper-formula protocol or released-code protocol for Stage 1 masking.",
    )
    parser.add_argument(
        "--center-update-strategy",
        choices=("monotonic_discrete_search",),
        default="monotonic_discrete_search",
        help="Deterministic Stage 2 update with loss and distance safeguards.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "similarity", "centers", "train"),
        default="all",
    )
    parser.add_argument("--output-dir", default="./save", type=str)
    parser.add_argument(
        "--similarity-cache",
        type=str,
        default=None,
        help="Explicit audited Stage 1 cache to reuse after Stage 2-only changes.",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--parent-b1-config", default=None,
                        help="Completed NABirds-B B1 run config.json used by I2-NFD.")
    parser.add_argument("--center-cache", default=None,
                        help="Audited B1 center cache for this code length.")
    parser.add_argument("--teacher-cache", default=None,
                        help="Audited classifier cache from the B1 parent.")
    parser.add_argument("--nfd-alpha", type=float, default=0.75)
    parser.add_argument("--nfd-weight", type=float, default=0.01)
    parser.add_argument("--nfd-topk", type=int, default=5)
    parser.add_argument("--nfd-warmup", type=int, default=20)
    parser.add_argument("--nfd-ramp-end", type=int, default=40)
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic algorithms; not recommended for reported B0 runs.",
    )
    args = parser.parse_args()

    if args.val_per_class is None:
        args.val_per_class = 2 if args.dataset.lower() == "nabirds-b" else 10

    if args.variant == "b2nfd":
        from utils.neighbor_feature_distill import VERSION, nfd_weight
        if args.dataset.lower() != "nabirds-b":
            parser.error("--variant b2nfd is available only for NABirds-B")
        if not all((args.parent_b1_config, args.similarity_cache,
                    args.center_cache, args.teacher_cache)):
            parser.error("I2-NFD requires --parent-b1-config, --similarity-cache, "
                         "--center-cache, and --teacher-cache")
        if args.force_recompute or args.stage not in ("all", "train"):
            parser.error("I2-NFD reuses frozen B1 artifacts and supports only --stage all/train")
        try:
            nfd_weight(1, args.nfd_weight, args.nfd_warmup, args.nfd_ramp_end)
            if not math.isfinite(args.nfd_alpha) or not 0 <= args.nfd_alpha < 1:
                raise ValueError("--nfd-alpha must be finite in [0, 1)")
            if not 0 < args.nfd_topk < args.num_classes:
                raise ValueError("--nfd-topk must be between 1 and num-classes - 1")
            if args.epoch < args.nfd_ramp_end:
                raise ValueError("--epoch must cover the full NFD weight ramp")
        except ValueError as error:
            parser.error(str(error))
        args.nfd_version = VERSION
        args.nfd_feature_dim = 512
        args.nfd_teacher_policy = "frozen-eval-same-augmented-input"
        args.nfd_projection_policy = "i1-topk-prototype-differences-svd-trace-normalized"
    elif any((args.parent_b1_config, args.center_cache, args.teacher_cache)):
        parser.error("Parent B1, center, and teacher cache options require --variant b2nfd")

    for name in ("relation_kappa", "relation_temperature", "relation_weight_floor"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.relation_weight_floor > 1:
        parser.error("--relation-weight-floor must be <= 1")
    if args.variant == "b0" and (
        args.relation_mode != "full" or args.relation_kappa != 30.0
        or args.relation_temperature != 1.0 or args.relation_weight_floor != 0.05
    ):
        parser.error("Relation options require --variant b1/b2nfd")
    if args.info == "[SHC-B0]":
        args.info = f"[SHC-{args.variant.upper()}]"

    if args.force_recompute and args.similarity_cache:
        parser.error("--force-recompute and --similarity-cache cannot be used together")

    if len(args.topK) != 3:
        parser.error("--topK must contain exactly: -1 100 1000")
    if args.topK != [-1, 100, 1000]:
        parser.error("B0 reporting order is fixed to --topK -1 100 1000")
    if args.cpu:
        args.device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; pass --cpu only for diagnostics")
        args.device = torch.device(f"cuda:{args.gpu}")
    args.net = ResNet
    args.git_commit = current_git_commit()
    args.source_sha256 = source_sha256()
    return args


def main():
    args = load_config()
    seed_everything(args.seed, deterministic=not args.non_deterministic)

    (
        train_loader,
        relation_loader,
        val_loader,
        query_loader,
        database_loader,
        _,
        _,
        _,
        num_database,
    ) = load_data(args)

    nfd = None
    centers = None
    if args.variant == "b2nfd":
        from utils.frozen_b1 import load_frozen_b1
        from utils.neighbor_feature_distill import load_nfd

        _, centers = load_frozen_b1(args)
        nfd = load_nfd(args)

    run_config = {
        "dataset": args.dataset,
        "seed": args.seed,
        "split_sha256": args.split_hash,
        "code_length": args.code_length,
        "mask_strategy": args.mask_strategy,
        "center_update_strategy": args.center_update_strategy,
        "similarity_cache": args.similarity_cache,
        "classification_epochs": args.classify_epoch,
        "hash_epochs": args.epoch,
        "learning_rate": args.lr,
        "validation_per_class": args.val_per_class,
        "git_commit": args.git_commit,
    }
    if args.variant in ("b1", "b2nfd"):
        from utils.reliable_relation import relation_config
        run_config["variant"] = args.variant
        run_config["relation"] = relation_config(args)
    if args.paired_stage_seeds:
        run_config["paired_stage_seeds"] = True
    if args.variant in ("b1", "b2nfd") or args.paired_stage_seeds:
        run_config.update({
            "source_sha256": args.source_sha256,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "test_map": args.test_map,
            "resize_size": args.resize_size,
            "crop_size": args.crop_size,
            "beta": args.beta,
            "lambd": args.lambd,
            "deterministic": not args.non_deterministic,
        })
    if nfd is not None:
        run_config.update({
            "parent_b1_config_sha256": args.parent_b1_config_sha256,
            "parent_b1_git_commit": args.parent_b1_git_commit,
            "similarity_content_sha256": args.similarity_content_sha256,
            "center_content_sha256": args.center_content_sha256,
            "nfd_version": args.nfd_version,
            "nfd_alpha": args.nfd_alpha,
            "nfd_weight": args.nfd_weight,
            "nfd_topk": args.nfd_topk,
            "nfd_warmup": args.nfd_warmup,
            "nfd_ramp_end": args.nfd_ramp_end,
            "nfd_teacher_sha256": args.nfd_teacher_sha256,
            "nfd_basis_sha256": args.nfd_basis_sha256,
            "nfd_implementation_sha256": args.nfd_implementation_sha256,
        })
    args.run_id = stable_config_hash(run_config)
    args.run_dir = os.path.join(
        args.output_dir,
        "runs",
        f"{args.dataset}_{args.variant.upper()}_{args.code_length}bit_seed{args.seed}_{args.run_id}",
    )
    if nfd is not None and os.path.exists(args.run_dir):
        raise FileExistsError(f"I2-NFD run already exists: {args.run_dir}")
    write_run_manifest(args)

    if nfd is not None:
        torch.save(
            {"identity": nfd.identity, "basis": nfd.basis.cpu().clone(),
             "neighbors": nfd.neighbors.cpu().clone(), "ranks": nfd.ranks.cpu().clone()},
            os.path.join(args.run_dir, "nfd_projection.pt"),
        )
        print(f"Loaded frozen NABirds-B B1 artifacts for I2-NFD: {args.parent_b1_config}")

    if nfd is None:
        similarity = GenerateSimilarityMatrix(
            args, train_loader, relation_loader, val_loader
        )
    write_run_manifest(args)
    if args.stage == "similarity":
        return

    if nfd is None:
        centers = GenerateSemanticHashCenters(args, similarity.to(args.device))
    from utils.reliable_relation import tensor_sha256
    args.center_content_sha256 = tensor_sha256(centers)
    write_run_manifest(args)
    if args.stage == "centers":
        return

    if args.paired_stage_seeds:
        seed_everything(args.seed, deterministic=not args.non_deterministic)
        train_loader.generator.manual_seed(args.seed)
    train_val(
        args,
        centers,
        train_loader,
        val_loader,
        query_loader,
        database_loader,
        num_database,
        nfd=nfd,
    )


if __name__ == "__main__":
    main()
