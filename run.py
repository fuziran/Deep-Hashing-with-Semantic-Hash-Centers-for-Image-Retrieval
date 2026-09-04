import argparse
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
    write_run_manifest,
)


def load_config():
    parser = argparse.ArgumentParser(description="SHC paper-protocol reproduction")
    parser.add_argument(
        "--protocol",
        choices=("paper_repro", "audited_b0"),
        default="paper_repro",
        help="Published 10k/query-oracle protocol or the leakage-free audited B0.",
    )
    parser.add_argument("--seed", default=60, type=int)
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
    parser.add_argument("--val-per-class", default=0, type=int)
    parser.add_argument(
        "--mask-strategy",
        choices=("predicted_argmax", "ground_truth"),
        default="predicted_argmax",
        help="Paper-formula protocol or released-code protocol for Stage 1 masking.",
    )
    parser.add_argument(
        "--center-update-strategy",
        choices=("paper_projected_gradient", "monotonic_discrete_search"),
        default="paper_projected_gradient",
        help="Paper Algorithm 1 update or the repaired audited B0 update.",
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
    args = parser.parse_args()

    if args.force_recompute and args.similarity_cache:
        parser.error("--force-recompute and --similarity-cache cannot be used together")

    if args.protocol == "paper_repro" and args.val_per_class != 0:
        parser.error("paper_repro requires --val-per-class 0 (10,000 training images)")
    if args.protocol == "audited_b0" and args.val_per_class <= 0:
        parser.error("audited_b0 requires a positive --val-per-class")

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
        "protocol": args.protocol,
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
    args.run_id = stable_config_hash(run_config)
    args.run_dir = os.path.join(
        args.output_dir,
        "runs",
        f"{args.dataset}_{args.protocol}_{args.code_length}bit_seed{args.seed}_{args.run_id}",
    )
    write_run_manifest(args)

    similarity = GenerateSimilarityMatrix(
        args, train_loader, relation_loader, val_loader
    )
    if args.stage == "similarity":
        return

    centers = GenerateSemanticHashCenters(args, similarity.to(args.device))
    if args.stage == "centers":
        return

    train_val(
        args,
        centers,
        train_loader,
        val_loader,
        query_loader,
        database_loader,
        num_database,
    )


if __name__ == "__main__":
    main()
