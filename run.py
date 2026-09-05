import argparse
import os

# Must be set before the first CUDA BLAS workspace is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from GenerateSemanticHashCenters import GenerateSemanticHashCenters
from GenerateSimilarityMatrix import GenerateSimilarityMatrix
from data.data_loader import load_data
from network import ResNet
from train import evaluate_query_checkpoint, train_val
from utils.experiment import (
    current_git_commit,
    seed_everything,
    stable_config_hash,
    write_run_manifest,
)


def load_config():
    parser = argparse.ArgumentParser(description="Auditable SHC B0 and RSM B1 experiments")
    parser.add_argument("--seed", default=60, type=int)
    parser.add_argument("--info", default=None, type=str)
    parser.add_argument(
        "--method", choices=("B0", "A1", "A2", "A3", "B1"), default="B0"
    )
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
    parser.add_argument("--rsm-views", default=4, type=int)
    parser.add_argument(
        "--rsm-temperature-grid",
        nargs="+",
        default=[0.7, 1.0, 1.5, 2.0],
        type=float,
    )
    parser.add_argument("--rsm-confusion-alpha", default=0.7, type=float)
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
        choices=("all", "similarity", "centers", "train", "evaluate"),
        default="all",
    )
    parser.add_argument("--output-dir", default="./save", type=str)
    parser.add_argument(
        "--classifier-cache",
        type=str,
        default=None,
        help="Explicit audited B0 classifier cache reused by A1/A2/A3/B1.",
    )
    parser.add_argument(
        "--similarity-cache",
        type=str,
        default=None,
        help="Explicit audited Stage 1 cache to reuse after Stage 2-only changes.",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--evaluate-query",
        action="store_true",
        help="Evaluate the frozen checkpoint once after training; omit during development.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Frozen best_model_state.pt used with --stage evaluate.",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic algorithms; not recommended for reported B0 runs.",
    )
    args = parser.parse_args()

    if args.force_recompute and args.similarity_cache:
        parser.error("--force-recompute and --similarity-cache cannot be used together")
    if args.method != "B0" and args.stage != "evaluate" and not args.classifier_cache:
        parser.error("A1/A2/A3/B1 require the audited B0 --classifier-cache")
    if args.stage == "evaluate" and not args.checkpoint:
        parser.error("--stage evaluate requires --checkpoint")
    if args.stage != "evaluate" and args.checkpoint:
        parser.error("--checkpoint is only valid with --stage evaluate")
    if args.rsm_views < 1:
        parser.error("--rsm-views must be at least one")
    if not 0.0 <= args.rsm_confusion_alpha <= 1.0:
        parser.error("--rsm-confusion-alpha must be in [0, 1]")
    if any(value <= 0 for value in args.rsm_temperature_grid):
        parser.error("all --rsm-temperature-grid values must be positive")
    if not any(abs(value - 1.0) < 1e-12 for value in args.rsm_temperature_grid):
        parser.error("--rsm-temperature-grid must include 1.0")

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
    if args.info is None:
        args.info = f"[SHC-{args.method}]"
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

    if args.stage == "evaluate":
        evaluate_query_checkpoint(
            args,
            args.checkpoint,
            query_loader,
            database_loader,
            num_database,
        )
        return

    run_config = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "split_sha256": args.split_hash,
        "code_length": args.code_length,
        "mask_strategy": args.mask_strategy,
        "center_update_strategy": args.center_update_strategy,
        "similarity_cache": args.similarity_cache,
        "classifier_cache": args.classifier_cache,
        "classification_epochs": args.classify_epoch,
        "hash_epochs": args.epoch,
        "learning_rate": args.lr,
        "validation_per_class": args.val_per_class,
        "rsm_views": args.rsm_views,
        "rsm_temperature_grid": args.rsm_temperature_grid,
        "rsm_confusion_alpha": args.rsm_confusion_alpha,
        "git_commit": args.git_commit,
    }
    args.run_id = stable_config_hash(run_config)
    args.run_dir = os.path.join(
        args.output_dir,
        "runs",
        f"{args.dataset}_{args.method}_{args.code_length}bit_seed{args.seed}_{args.run_id}",
    )
    write_run_manifest(args)

    similarity = GenerateSimilarityMatrix(
        args, train_loader, relation_loader, val_loader
    )
    write_run_manifest(args)
    if args.stage == "similarity":
        return

    # Stage-local reset prevents classifier loading/inference from changing center RNG.
    seed_everything(args.seed, deterministic=not args.non_deterministic)
    centers = GenerateSemanticHashCenters(args, similarity.to(args.device))
    if args.stage == "centers":
        return

    # Keep hash-network initialization paired across B0/A1/A2/A3/B1.
    seed_everything(args.seed, deterministic=not args.non_deterministic)
    if getattr(train_loader, "generator", None) is not None:
        train_loader.generator.manual_seed(args.seed)
    train_val(
        args,
        centers,
        train_loader,
        val_loader,
        database_loader,
        num_database,
    )
    if args.evaluate_query:
        evaluate_query_checkpoint(
            args,
            os.path.join(args.run_dir, "best_model_state.pt"),
            query_loader,
            database_loader,
            num_database,
        )


if __name__ == "__main__":
    main()
