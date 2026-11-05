"""Three B0-only Cars-B experiments using the existing audited SHC objective."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
VERSION = "stanford-cars-b-audited-b0-v1"
METRICS = ("mAP@ALL", "mAP@100", "mAP@1000")
PAPER_COUNTS = dict(train=5880, query=1958, database=8347)


def configuration(root, gpu=0):
    return dict(dataset="Stanford_Cars-B", variant="b0", root=str(root), num_classes=196,
        seed=60, lr=1e-4, classify_epoch=300, epoch=300, test_map=5, beta=1., lambd=1e-4,
        batch_size=64, num_workers=0, resize_size=256, crop_size=224, val_per_class=2,
        mask_strategy="predicted_argmax", center_update_strategy="monotonic_discrete_search", gpu=gpu)


def training_command(settings, bit, output, digest):
    command = [sys.executable, str(REPO / "run.py")]
    for key, value in {**settings, "code_length": bit, "output_dir": str(output),
                       "expected_dataset_sha256": digest, "info": f"[Cars-B-B0-{bit}bit]"}.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command + ["--paired-stage-seeds", "--topK", "-1", "100", "1000"]


def preflight(options):
    print("Checking Cars-B configuration and image contents; no image loaders are iterated.", flush=True)
    import torch
    from data.data_loader import load_data
    from utils.experiment import file_sha256, source_sha256
    output = Path(options.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    settings = configuration(Path(options.root).resolve(), options.gpu)
    args = argparse.Namespace(**settings, device=torch.device("cpu"))
    with tempfile.TemporaryDirectory(prefix="cars_b0_check_") as scratch:
        args.output_dir = scratch
        loaders = load_data(args)
        del loaders
    if args.source_counts != PAPER_COUNTS:
        raise ValueError(f"Expected curated Cars-B counts {PAPER_COUNTS}, got {args.source_counts}; do not use Cars-A or resample query/database")
    weights = REPO / "models_ckpt" / "resnet34-b627a593.pth"
    if not weights.is_file():
        weights = Path(torch.hub.get_dir()) / "checkpoints" / weights.name
    if not weights.is_file():
        raise FileNotFoundError("ImageNet resnet34-b627a593.pth is missing; prepare it once as described in README_STANFORD_CARS_B0.md")
    return dict(version=VERSION, settings=settings, source_sha256=source_sha256(),
        launcher_sha256=file_sha256(__file__), split_hash=args.split_hash,
        dataset_content_sha256=args.dataset_content_sha256, counts=args.dataset_counts,
        source_counts=args.source_counts, exact_content_duplicates=getattr(args,"exact_content_duplicates",[]),
        pretrained_path=str(weights.resolve()),
        pretrained_sha256=file_sha256(weights), items=[dict(bits=bit,
            command=training_command(settings, bit, output, args.dataset_content_sha256)) for bit in (16,32,64)])


def report(output):
    from utils.experiment import file_sha256, source_sha256
    plan = json.loads((output / "experiment_plan.json").read_text(encoding="utf-8"))
    if plan["version"] != VERSION or plan["source_sha256"] != source_sha256():
        raise ValueError("Use the source checkout that trained these B0 experiments")
    rows = []
    for bit in (16,32,64):
        runs = list((output / "runs").glob(f"Stanford_Cars-B_B0_{bit}bit_seed60_*"))
        if len(runs) != 1:
            raise ValueError(f"Expected exactly one completed {bit}-bit run; training may still be running")
        run = runs[0]
        read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
        config, metrics, history = read("config.json"), read("metrics.json"), read("training_history.json")
        if len(history) != 300 or [r["epoch"] for r in history] != list(range(1,301)):
            raise ValueError(f"Incomplete {bit}-bit training")
        for key, value in {**plan["settings"], "code_length":bit, "source_sha256":plan["source_sha256"],
                "split_hash":plan["split_hash"], "dataset_content_sha256":plan["dataset_content_sha256"],
                "paired_stage_seeds":True}.items():
            if config.get(key) != value:
                raise ValueError(f"{bit}-bit protocol mismatch: {key}")
        checked = [r for r in history if "validation" in r]
        if not checked or any(not math.isfinite(r["validation"][k]) for r in checked for k in METRICS):
            raise ValueError("Invalid validation scores")
        best = max(checked, key=lambda r:r["validation"]["mAP@ALL"])
        if (metrics["selected_epoch"] != best["epoch"] or metrics["selection_metric"] != "validation mAP@ALL"
                or metrics["best_validation_mAP@ALL"] != best["validation"]["mAP@ALL"]
                or metrics.get("query_evaluated") is not True
                or metrics["counts"] != {k:plan["counts"][k] for k in ("query","database")}):
            raise ValueError("Invalid selected model or query protocol")
        for key, filename in (("checkpoint_sha256","best_model_state.pt"), ("training_config_sha256","config.json"),
                ("training_history_sha256","training_history.json"), ("retrieval_outputs_sha256","retrieval_outputs.pt")):
            if metrics.get(key) != file_sha256(run / filename):
                raise ValueError(f"Saved artifact mismatch: {filename}")
        values = metrics["query_metrics"]
        if any(not isinstance(values[k], (int,float)) or not math.isfinite(values[k]) or not 0<=values[k]<=1 for k in METRICS):
            raise ValueError("Invalid query scores")
        rows.append(dict(bits=bit, selected_epoch=metrics["selected_epoch"], run_dir=str(run), **values))
    result = dict(dataset="Stanford_Cars-B", variant="b0", seed=60, completed_epochs=300,
        query_evaluated=True, counts=plan["counts"], results=rows,
        exact_cross_split_content_duplicate_pairs=len(plan.get("exact_content_duplicates",[])))
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--output-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", help="Read completed query results; never retrain or reevaluate")
    options = parser.parse_args(argv)
    try:
        if options.report:
            print(json.dumps(report(Path(options.report).resolve()), indent=2), flush=True)
            return
        if not options.root or not options.output_root:
            raise ValueError("--root and --output-root are required")
        if not options.dry_run:
            print("1/4 Checking SHC environment and GPU", flush=True)
            import torch
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable in the selected environment")
            print(torch.__version__, torch.cuda.get_device_name(options.gpu), flush=True)
            assert (torch.ones(1, device=f"cuda:{options.gpu}")+1).item() == 2
            print("2/4 Running baseline regression tests", flush=True)
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=REPO, check=True)
        print("3/4 Checking data and ImageNet initialization", flush=True)
        plan = preflight(options)
        print(json.dumps({k:plan[k] for k in ("source_counts","counts","split_hash","dataset_content_sha256")}, indent=2), flush=True)
        if options.dry_run:
            print("Preflight passed; no classifier/hash model trained, no final query evaluated.", flush=True)
            return
        from utils.experiment import file_sha256, source_sha256
        output = Path(options.output_root).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print("4/4 Training B0 16 -> 32 -> 64 bit; final query is evaluated once per selected model", flush=True)
        for item in plan["items"]:
            if (source_sha256() != plan["source_sha256"] or file_sha256(__file__) != plan["launcher_sha256"]
                    or file_sha256(plan["pretrained_path"]) != plan["pretrained_sha256"]):
                raise ValueError("Source or pretrained initialization changed during the experiment")
            log = output / f"{item['bits']}bit.log"
            print(f"Starting {item['bits']}bit B0; log: {log}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(item["command"], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True,
                               env={**os.environ, "PYTHONUNBUFFERED":"1"})
        print(json.dumps(report(output), indent=2), flush=True)
        print("All three B0 training/query experiments completed.", flush=True)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Stanford Cars-B B0 failed: {error}\n")


if __name__ == "__main__":
    print("Stanford Cars-B B0 launcher started", flush=True)
    main()
