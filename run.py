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
    parser = argparse.ArgumentParser(description="Faithful and auditable SHC B0 baseline")
    parser.add_argument("--seed", default=60, type=int)
    parser.add_argument("--variant", choices=("b0", "b1", "b2", "b2r", "b2m"), default="b0")
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
    parser.add_argument("--val-per-class", default=10, type=int)
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
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic algorithms; not recommended for reported B0 runs.",
    )
    parser.add_argument("--parent-b1-config", default=None)
    parser.add_argument("--center-cache", default=None)
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--comp-weight", type=float, default=0.1)
    parser.add_argument("--comp-temperature", type=float, default=0.1)
    parser.add_argument("--comp-m0", type=float, default=0.1)
    parser.add_argument("--comp-m1", type=float, default=0.4)
    parser.add_argument("--comp-warmup", type=int, default=20)
    parser.add_argument("--comp-ramp-end", type=int, default=40)
    parser.add_argument("--i2r-rho", type=float, default=0.25)
    parser.add_argument("--i2r-stats-start", type=int, default=16)
    parser.add_argument("--i2r-warmup", type=int, default=20)
    parser.add_argument("--i2r-ramp-end", type=int, default=40)
    parser.add_argument("--mix-mode", choices=("semantic", "plain"), default="semantic")
    parser.add_argument("--mix-weight", type=float, default=0.1)
    parser.add_argument("--mix-budget", type=float, default=None,
                        help="Soft-target Hamming budget; default code-length/16.")
    parser.add_argument("--mix-warmup", type=int, default=20)
    parser.add_argument("--mix-ramp-end", type=int, default=40)
    args = parser.parse_args()
    if args.variant == "b2m":
        from utils.semantic_mixup import VERSION, mix_weight
        if args.mix_budget is None:
            args.mix_budget = args.code_length / 16
        try:
            mix_weight(1, args.mix_weight, args.mix_warmup, args.mix_ramp_end)
            if not math.isfinite(args.mix_budget) or args.mix_budget <= 0:
                raise ValueError("Mixup budget must be finite and positive")
            if args.epoch < args.mix_ramp_end:
                raise ValueError("B2M epoch budget must cover the complete weight ramp")
        except ValueError as error:
            parser.error(str(error))
        args.mix_version = VERSION
        args.mix_seed = args.seed + 200003
        args.mix_bn_policy = "clean_running_stats"
    if args.variant == "b2r":
        from utils.bit_supervision import VERSION, risk_schedule
        try:
            risk_schedule(1, args.i2r_rho, args.i2r_stats_start,
                          args.i2r_warmup, args.i2r_ramp_end)
            if args.epoch < args.i2r_ramp_end:
                raise ValueError("B2R epoch budget must cover the complete weight ramp")
        except ValueError as error:
            parser.error(str(error))
        args.i2r_version = VERSION
    if args.variant in ("b2", "b2r", "b2m"):
        if not all((args.parent_b1_config, args.similarity_cache, args.center_cache)):
            parser.error("B2/B2R/B2M requires --parent-b1-config, --similarity-cache and --center-cache")
        if not args.paired_stage_seeds or args.force_recompute:
            parser.error("B2/B2R/B2M requires --paired-stage-seeds and forbids --force-recompute")
        if args.stage not in ("all", "train"):
            parser.error("B2/B2R/B2M supports only --stage all/train with frozen B1 artifacts")
    elif args.parent_b1_config or args.center_cache:
        parser.error("Parent B1 and center cache options require --variant b2/b2r/b2m")
    from utils.competition import competition_weight
    try:
        competition_weight(1, args.comp_weight, args.comp_warmup, args.comp_ramp_end)
        if (not all(math.isfinite(x) for x in (args.comp_temperature, args.comp_m0, args.comp_m1))
                or args.comp_temperature <= 0 or min(args.comp_m0, args.comp_m1) < 0
                or args.comp_m0 + args.comp_m1 > 1):
            raise ValueError("Require finite temperature > 0, m0/m1 >= 0 and m0+m1 <= 1")
    except ValueError as error:
        parser.error(str(error))

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
        parser.error("Relation options require --variant b1/b2/b2r/b2m")
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
    if args.variant in ("b1", "b2", "b2r", "b2m"):
        from utils.reliable_relation import relation_config
        run_config["variant"] = args.variant
        run_config["relation"] = relation_config(args)
    if args.paired_stage_seeds:
        run_config["paired_stage_seeds"] = True
    if args.variant in ("b1", "b2", "b2r", "b2m") or args.paired_stage_seeds:
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
    if args.skip_query:
        run_config["skip_query"] = True
    if args.variant in ("b2", "b2r", "b2m"):
        from utils.frozen_b1 import load_frozen_b1
        similarity, centers = load_frozen_b1(args)
        run_config.update({key: value for key, value in vars(args).items()
                           if key.startswith("parent_")
                           or key.startswith({"b2": "comp_", "b2r": "i2r_", "b2m": "mix_"}[args.variant])})
        run_config.update({
            "center_content_sha256": args.center_content_sha256,
            "similarity_content_sha256": args.similarity_content_sha256,
        })
    args.run_id = stable_config_hash(run_config)
    args.run_dir = os.path.join(
        args.output_dir,
        "runs",
        f"{args.dataset}_{args.variant.upper()}_{args.code_length}bit_seed{args.seed}_{args.run_id}",
    )
    if args.variant in ("b2", "b2r", "b2m") and os.path.exists(args.run_dir):
        raise FileExistsError(f"Run already exists; choose a new --output-dir: {args.run_dir}")
    write_run_manifest(args)

    if args.variant not in ("b2", "b2r", "b2m"):
        similarity = GenerateSimilarityMatrix(
            args, train_loader, relation_loader, val_loader
        )
    write_run_manifest(args)
    if args.stage == "similarity":
        return

    if args.variant not in ("b2", "b2r", "b2m"):
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
        similarity=similarity if args.variant in ("b2", "b2r", "b2m") else None,
    )


if __name__ == "__main__":
    main()
