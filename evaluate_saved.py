"""Final query evaluation of an already selected checkpoint; never trains."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from data.data_loader import load_data
from network import ResNet
from train import _evaluate
from utils.coarse_aux import file_sha256
from utils.experiment import seed_everything, source_sha256
from utils.tools import CalcTopMapPerQuery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--root", help="Optional relocated CIFAR-100 raw directory")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    cli = parser.parse_args()
    run_dir = Path(cli.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    selected = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if selected.get("query_evaluated"):
        parser.error("This run already evaluated final queries; use its saved results")
    if config.get("source_sha256") != source_sha256():
        parser.error("Use the same source revision as the selected training run")
    checkpoint = run_dir / "best_model_state.pt"
    if selected.get("model_sha256") != file_sha256(checkpoint):
        parser.error("Selected checkpoint content differs from the recorded SHA256")
    destination = run_dir / "final_query"
    if destination.exists():
        parser.error(f"Final query output already exists: {destination}")
    args = SimpleNamespace(**config)
    args.device = torch.device("cpu" if cli.cpu else f"cuda:{cli.gpu}")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; --cpu is intended for diagnostics only")
    if cli.root:
        args.root = cli.root
    args.output_dir = str(destination)
    seed_everything(args.seed, deterministic=not args.non_deterministic)
    expected_split = args.split_hash
    loaders = load_data(args)
    if args.split_hash != expected_split:
        raise RuntimeError("Data split differs from the selected training run")
    net = ResNet(args.code_length).to(args.device)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    metrics, pr, qb, ql, db, dl = _evaluate(loaders[3], loaders[4], net, args, loaders[8])
    per_query = CalcTopMapPerQuery(qb.numpy(), ql.numpy(), db.numpy(), dl.numpy(), args.topK)
    result = {
        "selected_epoch": selected["selected_epoch"],
        "selection_metric": selected["selection_metric"],
        "model_sha256": selected["model_sha256"],
        "split_sha256": args.split_hash,
        "query_metrics": metrics,
    }
    (destination / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"query_binary": qb, "query_label": ql,
                "database_binary": db, "database_label": dl,
                "pr_data": pr, "per_query_ap": per_query}, destination / "retrieval_outputs.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
