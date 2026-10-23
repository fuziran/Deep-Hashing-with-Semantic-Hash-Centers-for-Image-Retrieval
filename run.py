import argparse
import json
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
    parser.add_argument("--variant", choices=("b0", "b1", "b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd"), default="b0")
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
    parser.add_argument("--rank-mode", choices=("semantic", "uniform"), default="semantic")
    parser.add_argument("--rank-weight", type=float, default=0.05)
    parser.add_argument("--rank-capacity", type=int, default=1024)
    parser.add_argument("--rank-max-age", type=int, default=16)
    parser.add_argument("--rank-chunk-size", type=int, default=64)
    parser.add_argument("--rank-warmup", type=int, default=20)
    parser.add_argument("--rank-ramp-end", type=int, default=40)
    parser.add_argument("--resume-sbr", default=None,
                        help="Trusted SBR epoch checkpoint; requires an unused output directory.")
    parser.add_argument("--teacher-cache", default=None,
                        help="Audited classifier cache belonging to the frozen parent B1.")
    parser.add_argument("--nfd-alpha", type=float, default=.75,
                        help="Neighborhood weighting; 0 is the ordinary feature-distillation control.")
    parser.add_argument("--nfd-weight", type=float, default=.01)
    parser.add_argument("--nfd-topk", type=int, default=5)
    parser.add_argument("--nfd-warmup", type=int, default=20)
    parser.add_argument("--nfd-ramp-end", type=int, default=40)
    parser.add_argument("--npa-mode", choices=("protected", "covariance", "isotropic"), default="protected")
    parser.add_argument("--npa-rho", type=float, default=.05)
    parser.add_argument("--npa-shrinkage", type=float, default=.2)
    parser.add_argument("--npa-weight", type=float, default=1.)
    parser.add_argument("--npa-warmup", type=int, default=40)
    parser.add_argument("--npa-ramp-end", type=int, default=60)
    parser.add_argument("--fcd-cache", default=None)
    parser.add_argument("--fcd-mode", choices=("ce", "plain", "neighbor"), default="neighbor")
    parser.add_argument("--fcd-weight", type=float, default=.05)
    parser.add_argument("--fcd-temperature", type=float, default=2.)
    parser.add_argument("--fcd-gamma", type=float, default=1.)
    parser.add_argument("--fcd-topk", type=int, default=5)
    parser.add_argument("--fcd-warmup", type=int, default=40)
    parser.add_argument("--fcd-ramp-end", type=int, default=60)
    args = parser.parse_args()
    if args.variant == "b3fcd":
        from utils.foundation_distill import VERSION, ramp
        try:
            ramp(1,args.fcd_weight,args.fcd_warmup,args.fcd_ramp_end)
            if not math.isfinite(args.fcd_temperature) or args.fcd_temperature <= 0:
                raise ValueError("FCD temperature must be finite and positive")
            if not math.isfinite(args.fcd_gamma) or not 0 <= args.fcd_gamma <= 1:
                raise ValueError("FCD gamma must be finite in [0,1]")
            if not 0 < args.fcd_topk < args.num_classes:
                raise ValueError("FCD topk must be smaller than num-classes")
            if args.fcd_weight > 0 and (not args.fcd_cache or args.epoch < args.fcd_ramp_end):
                raise ValueError("Enabled FCD requires --fcd-cache and epoch budget covering ramp")
        except ValueError as error:
            parser.error(str(error))
        args.fcd_version=VERSION
    elif args.fcd_cache:
        parser.error("--fcd-cache requires --variant b3fcd")
    if args.variant == "b3npa":
        from utils.neighborhood_protected_augmentation import VERSION, npa_weight
        try:
            npa_weight(1, args.npa_weight, args.npa_warmup, args.npa_ramp_end)
            if not math.isfinite(args.npa_rho) or args.npa_rho < 0:
                raise ValueError("NPA rho must be finite and nonnegative")
            if not math.isfinite(args.npa_shrinkage) or not 0 <= args.npa_shrinkage <= 1:
                raise ValueError("NPA shrinkage must be finite in [0,1]")
            if args.npa_weight > 0 and args.npa_rho > 0 and args.epoch < args.npa_ramp_end:
                raise ValueError("NPA epoch budget must cover its weight ramp")
        except ValueError as error:
            parser.error(str(error))
        args.npa_version = VERSION
    if args.variant in ("b2nfd", "b3npa", "b3fcd"):
        from utils.neighbor_feature_distill import VERSION, nfd_weight
        try:
            nfd_weight(1, args.nfd_weight, args.nfd_warmup, args.nfd_ramp_end)
            if not math.isfinite(args.nfd_alpha) or not 0 <= args.nfd_alpha < 1:
                raise ValueError("NFD alpha must be finite in [0, 1)")
            if not 0 < args.nfd_topk < args.num_classes:
                raise ValueError("NFD topk must be positive and smaller than num-classes")
            if args.epoch < args.nfd_ramp_end:
                raise ValueError("NFD epoch budget must cover the complete weight ramp")
            if not args.teacher_cache:
                raise ValueError("NFD requires --teacher-cache from the same parent B1")
        except ValueError as error:
            parser.error(str(error))
        args.nfd_version = VERSION
        args.nfd_feature_dim = 512
        args.nfd_teacher_policy = "frozen-eval-same-augmented-input"
        args.nfd_projection_policy = "i1-topk-prototype-differences-svd-trace-normalized"
    elif args.teacher_cache:
        parser.error("--teacher-cache requires --variant b2nfd/b3npa")
    if args.resume_sbr and args.variant != "b2sbr":
        parser.error("--resume-sbr requires --variant b2sbr")
    if args.variant == "b2sbr":
        from utils.semantic_rank import VERSION, rank_weight
        try:
            rank_weight(1, args.rank_weight, args.rank_warmup, args.rank_ramp_end)
            if min(args.rank_capacity, args.rank_max_age, args.rank_chunk_size, args.code_length) <= 0:
                raise ValueError("Ranking capacity/age/chunk/bits must be positive")
            if args.epoch < args.rank_ramp_end:
                raise ValueError("SBR epoch budget must cover the complete weight ramp")
        except ValueError as error:
            parser.error(str(error))
        args.rank_version = VERSION
        args.rank_margin_bits = 1
        args.rank_topk = 5
        args.rank_semantic_fraction = 0.5 if args.rank_mode == "semantic" else 0.0
        args.rank_queue_policy = "post-step-pre-update-codes-no-warmup-queue"
        args.rank_proxy = "binary-forward-continuous-anchor-gradient"

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
    if args.variant in ("b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd"):
        if not all((args.parent_b1_config, args.similarity_cache, args.center_cache)):
            parser.error("B2/B2R/B2M/B2SBR/B2NFD requires --parent-b1-config, --similarity-cache and --center-cache")
        if not args.paired_stage_seeds or args.force_recompute:
            parser.error("B2/B2R/B2M/B2SBR/B2NFD requires --paired-stage-seeds and forbids --force-recompute")
        if args.stage not in ("all", "train"):
            parser.error("B2/B2R/B2M/B2SBR/B2NFD supports only --stage all/train with frozen B1 artifacts")
    elif args.parent_b1_config or args.center_cache:
        parser.error("Parent B1 and center cache options require --variant b2/b2r/b2m/b2sbr")
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
        parser.error("Relation options require --variant b1/b2/b2r/b2m/b2sbr/b2nfd")
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
    if args.variant in ("b1", "b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd"):
        from utils.reliable_relation import relation_config
        run_config["variant"] = args.variant
        run_config["relation"] = relation_config(args)
    if args.paired_stage_seeds:
        run_config["paired_stage_seeds"] = True
    if args.variant in ("b1", "b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd") or args.paired_stage_seeds:
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
    nfd = None
    npa = None
    fcd = None
    if args.skip_query:
        run_config["skip_query"] = True
    if args.variant in ("b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd"):
        from utils.frozen_b1 import load_frozen_b1
        similarity, centers = load_frozen_b1(args)
        if args.variant in ("b2nfd", "b3npa", "b3fcd"):
            from utils.neighbor_feature_distill import load_nfd
            nfd = load_nfd(args)
        if args.variant == "b3npa":
            from utils.neighborhood_protected_augmentation import prepare_npa
            npa = prepare_npa(args, nfd, relation_loader, train_loader)
        if args.variant == "b3fcd":
            from utils.foundation_distill import load_fcd
            fcd = load_fcd(args, centers, similarity, train_loader.dataset)
        run_config.update({key: value for key, value in vars(args).items()
                           if key.startswith("parent_")
                           or key.startswith({"b2": "comp_", "b2r": "i2r_", "b2m": "mix_", "b2sbr": "rank_", "b2nfd": "nfd_", "b3npa": "npa_", "b3fcd": "fcd_"}[args.variant])
                           or (args.variant in ("b3npa", "b3fcd") and key.startswith("nfd_"))})
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
    if args.variant in ("b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd") and os.path.exists(args.run_dir):
        raise FileExistsError(f"Run already exists; choose a new --output-dir: {args.run_dir}")
    if args.variant == "b2sbr":
        print("SBR configuration=" + json.dumps({
            key: value for key, value in vars(args).items()
            if key.startswith("rank_") or key in (
                "variant", "seed", "code_length", "run_id", "git_commit", "source_sha256",
                "split_hash", "parent_b1_run_id", "similarity_content_sha256",
                "center_content_sha256", "skip_query", "resume_sbr")
        }, sort_keys=True))
    write_run_manifest(args)
    if npa is not None:
        torch.save(npa.artifact, os.path.join(args.run_dir, "npa_statistics.pt"))
        del npa.artifact
        print("NPA configuration=" + json.dumps({key: value for key, value in vars(args).items()
                                                if key.startswith("npa_")}))
    if nfd is not None:
        torch.save({"identity": nfd.identity, "basis": nfd.basis.cpu().clone(),
                    "neighbors": nfd.neighbors.cpu().clone(), "ranks": nfd.ranks.cpu().clone()},
                   os.path.join(args.run_dir, "nfd_projection.pt"))
        print("NFD configuration=" + json.dumps(nfd.identity))

    if args.variant not in ("b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd"):
        similarity = GenerateSimilarityMatrix(
            args, train_loader, relation_loader, val_loader
        )
    write_run_manifest(args)
    if args.stage == "similarity":
        return

    if args.variant not in ("b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd"):
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
        similarity=similarity if args.variant in ("b2", "b2r", "b2m", "b2sbr", "b2nfd", "b3npa", "b3fcd") else None,
        nfd=nfd,
        npa=npa,
        fcd=fcd,
    )


if __name__ == "__main__":
    main()
