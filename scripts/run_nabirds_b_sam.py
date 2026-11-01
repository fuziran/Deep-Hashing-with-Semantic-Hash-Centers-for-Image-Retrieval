"""NABirds-B SAM-only transfer of the fixed CIFAR recipe, with historical controls."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.experiment import file_sha256, source_sha256

VERSION = "nabirds-b-nfd-sam-v1"
RHO = .05
METRICS = ("mAP@ALL", "mAP@100", "mAP@1000")
SHARED = ("dataset", "root", "seed", "num_classes", "code_length", "lr", "epoch", "classify_epoch",
          "test_map", "beta", "lambd", "batch_size", "num_workers", "resize_size", "crop_size",
          "val_per_class", "mask_strategy", "center_update_strategy", "relation_mode", "relation_kappa",
          "relation_temperature", "relation_weight_floor", "nfd_alpha", "nfd_weight", "nfd_topk", "nfd_warmup", "nfd_ramp_end")
ARTIFACTS = ("parent_b1_config", "similarity_cache", "center_cache", "teacher_cache")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_file(value, reference):
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [REPO / raw, reference.parent / raw]
    found = {p.resolve() for p in candidates if p.is_file()}
    if len(found) != 1:
        raise ValueError(f"Missing or ambiguous original artifact: {value} (reference {reference})")
    return found.pop()


def selected_reference(path):
    metrics = read_json(path.parent / "metrics.json")
    history = read_json(path.parent / "training_history.json")
    if len(history) != 300 or [row["epoch"] for row in history] != list(range(1, 301)):
        raise ValueError(f"Original NFD training is incomplete: {path}")
    selected = max((row for row in history if "validation" in row), key=lambda row: row["validation"]["mAP@ALL"])
    if selected["epoch"] != metrics["selected_epoch"]:
        raise ValueError(f"Original NFD selection mismatch: {path}")
    for query_file in (path.parent / "metrics.json", path.parent / "final_query" / "metrics.json"):
        if not query_file.is_file():
            continue
        payload = read_json(query_file)
        query = payload.get("query_metrics")
        # Original NABirds NFD predates query_evaluated/selection receipts.
        if isinstance(query, dict) and payload.get("query_evaluated") is not False:
            if any(not isinstance(query.get(k), (int, float)) or not math.isfinite(query[k])
                   or not 0 <= query[k] <= 1 for k in METRICS):
                raise ValueError(f"Invalid historical query results: {query_file}")
            return dict(validation=selected["validation"], selected_epoch=selected["epoch"],
                        query_metrics={k: query[k] for k in METRICS}, query_source=str(query_file),
                        query_source_sha256=file_sha256(query_file),
                        query_evaluated_flag_present="query_evaluated" in payload)
    raise ValueError(f"No saved original NFD query metrics found: {path.parent}; do not substitute validation scores")


def references(root, explicit, bits):
    paths = [Path(p).resolve() for p in explicit] if explicit else sorted(Path(root).resolve().rglob("config.json"))
    result = {}
    for path in paths:
        config = read_json(path)
        if (config.get("dataset", "").lower() != "nabirds-b" or config.get("variant") != "b2nfd"
                or config.get("seed") != 60 or config.get("code_length") not in bits
                or config.get("nfd_alpha") != .75 or config.get("nfd_weight") != .01
                or config.get("sam_rho", 0) != 0):
            if explicit:
                raise ValueError(f"Not a requested original NABirds-B NFD reference: {path}")
            continue
        bit = config["code_length"]
        if bit in result:
            raise ValueError(f"Multiple {bit}-bit references; pass one explicit --nfd-config per bit")
        result[bit] = path, config
    if set(result) != set(bits):
        raise ValueError(f"Missing NABirds-B NFD references for {sorted(set(bits) - set(result))} under {root}")
    return result


def build_command(path, config, output, gpu, digest):
    if (config.get("num_classes") != 555 or config.get("epoch") != 300
            or config.get("nfd_topk") != 5 or config.get("nfd_warmup") != 20
            or config.get("nfd_ramp_end") != 40 or config.get("val_per_class") != 2
            or config.get("non_deterministic") is not False or config.get("topK") != [-1, 100, 1000]
            or not isinstance(config.get("paired_stage_seeds"), bool)):
        raise ValueError(f"Unexpected original NABirds-B protocol: {path}")
    command = [sys.executable, str(REPO / "run.py"), "--variant", "b2nfd", "--skip-query",
               "--sam-rho", str(RHO), "--expected-dataset-sha256", digest,
               "--output-dir", str(output), "--gpu", str(gpu), "--topK", "-1", "100", "1000",
               "--info", f"[NABirds-B-NFD+SAM-{config['code_length']}bit]"]
    for key in SHARED:
        command += ["--" + key.replace("_", "-"), str(config[key])]
    for key in ARTIFACTS:
        command += ["--" + key.replace("_", "-"), str(resolve_file(config[key], path))]
    if config["paired_stage_seeds"]:
        command += ["--paired-stage-seeds"]
    return command


def make_plan(options):
    import torch
    from data.data_loader import load_data
    from utils.frozen_b1 import load_frozen_b1
    from utils.neighbor_feature_distill import load_nfd
    root = Path(options.output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Choose a new output root: {root}")
    selected = references(options.baseline_root, options.nfd_config, options.bits)
    first = selected[options.bits[0]][1]
    same_data = ("dataset", "root", "seed", "num_classes", "val_per_class", "split_hash")
    if any(config.get(key) != first.get(key) for _, config in selected.values() for key in same_data):
        raise ValueError("The selected original NFD runs use different data protocols")
    # One shared structural/content scan, no image loader iteration.
    args = argparse.Namespace(**first)
    args.device = torch.device("cpu")
    with tempfile.TemporaryDirectory(prefix="nabirds_sam_preflight_") as scratch:
        args.output_dir = scratch
        loaders = load_data(args)
        del loaders
    if args.split_hash != first["split_hash"]:
        raise ValueError("NABirds-B class paths/labels or train-validation split differ from the original NFD run")
    weights = REPO / "models_ckpt" / "resnet34-b627a593.pth"
    if not weights.is_file():
        weights = Path(torch.hub.get_dir()) / "checkpoints" / weights.name
    if not weights.is_file():
        raise FileNotFoundError("Existing resnet34-b627a593.pth required in models_ckpt or torch hub cache")
    items = []
    for bit in options.bits:
        path, config = selected[bit]
        output = root / f"{bit}bit"
        command = build_command(path, config, output, options.gpu, args.dataset_content_sha256)
        check_args = argparse.Namespace(**config)
        check_args.device = torch.device("cpu")
        for field in ARTIFACTS:
            setattr(check_args, field, str(resolve_file(config[field], path)))
        load_frozen_b1(check_args)
        nfd = load_nfd(check_args)
        for field in ("nfd_teacher_sha256", "nfd_basis_sha256", "center_content_sha256"):
            if getattr(check_args, field) != config[field]:
                raise ValueError(f"Original {bit}-bit artifact content mismatch: {field}")
        del nfd
        items.append(dict(bits=bit, reference_config=str(path), reference_config_sha256=file_sha256(path),
                          historical=selected_reference(path), output=str(output), command=command))
    return dict(version=VERSION, source_sha256=source_sha256(), launcher_sha256=file_sha256(__file__),
                bits=options.bits, seed=60, rho=.05, split_hash=args.split_hash,
                dataset_content_sha256=args.dataset_content_sha256, counts=args.dataset_counts,
                initialization_file=str(weights.resolve()), initialization_sha256=file_sha256(weights),
                cuda_available=torch.cuda.is_available(), items=items,
                note="Original split hash is preserved. Legacy runs lack a file-content fingerprint; this new digest binds current training/evaluation, not historical image bytes.")


def check_plan(root):
    plan = read_json(root / "sam_plan.json")
    if plan["version"] != VERSION or plan["source_sha256"] != source_sha256():
        raise ValueError("SAM plan/source mismatch; use the training checkout")
    return plan


def run_process(command, log):
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})


def training_report(root):
    from scripts.evaluate_nabirds_sam_query import validate_run
    plan = check_plan(root)
    rows = []
    for item in plan["items"]:
        runs = [p for p in (Path(item["output"]) / "runs").iterdir() if p.is_dir()]
        if len(runs) != 1:
            raise ValueError(f"Expected one completed {item['bits']}-bit SAM run")
        run = runs[0]
        config, metrics, values, _ = validate_run(run)
        if (config["sam_rho"] != .05 or config["code_length"] != item["bits"]
                or config["seed"] != 60 or config["epoch"] != 300
                or metrics["query_evaluated"] is not False):
            raise ValueError("SAM training protocol mismatch")
        for key in ("split_hash", "dataset_content_sha256"):
            if config[key] != plan[key]:
                raise ValueError(f"Training data differs from preflight: {key}")
        rows.append(dict(bits=item["bits"], run_dir=str(run), selected_epoch=metrics["selected_epoch"], **values))
    report = dict(seed=60, rho=.05, query_evaluated=False, comparison="historical-reference", results=rows)
    (root / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def smoke_test(command):
    import torch
    from unittest.mock import patch
    from run import load_config
    from data.data_loader import load_data
    from train import CSQLoss
    from utils.experiment import seed_everything
    from utils.frozen_b1 import load_frozen_b1
    from utils.neighbor_feature_distill import load_nfd
    from utils.sam_nfd import sam_nfd_step
    with patch.object(sys, "argv", command[1:]):
        args = load_config()
    with tempfile.TemporaryDirectory(prefix="nabirds_sam_smoke_") as scratch:
        args.output_dir = scratch
        seed_everything(args.seed)
        loaders = load_data(args)
        training = loaders[0]
        del loaders
        if args.dataset_content_sha256 != args.expected_dataset_sha256:
            raise ValueError("Dataset changed after preflight")
        _, centers = load_frozen_b1(args)
        nfd = load_nfd(args)
        seed_everything(args.seed)
        training.generator.manual_seed(args.seed)
        net = args.net(args.code_length).to(args.device).train()
        if centers.shape != (args.num_classes, args.code_length):
            centers = centers.T
        criterion = CSQLoss(args, args.code_length, centers.float())
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-5)
        images, labels, indices = next(iter(training))
        images, labels = images.to(args.device), labels.to(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        value, _ = sam_nfd_step(net, optimizer, criterion, nfd, images, labels, indices,
                                args, args.nfd_weight, sample_gradients=True)
        torch.cuda.synchronize(args.device)
        return dict(smoke_passed=True, bits=args.code_length, batch_size=len(images), clean_objective=value,
                    perturbation_norm=criterion.components["sam_perturbation_norm"],
                    peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device) / 2**30,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved(args.device) / 2**30,
                    query_evaluated=False, optimizer_updates=1)


def evaluate_queries(root, gpu):
    from scripts.evaluate_nabirds_sam_query import evaluate
    plan, trained = check_plan(root), training_report(root)
    output = root / "query"
    if output.exists():
        raise FileExistsError(f"Query output already exists: {output}; inspect existing results/logs")
    jobs = []
    for row in trained["results"]:
        destination = output / f"{row['bits']}bit"
        checked = evaluate(argparse.Namespace(run_dir=row["run_dir"], output_dir=str(destination),
                                             data_root=None, gpu=gpu, cpu=False, dry_run=True))
        if checked["dataset_content_sha256"] != plan["dataset_content_sha256"]:
            raise ValueError("Query data differs from the fixed training data")
        jobs.append(dict(bits=row["bits"], run_dir=row["run_dir"], destination=str(destination), command=[
            sys.executable, str(REPO / "scripts" / "evaluate_nabirds_sam_query.py"), "--run-dir", row["run_dir"],
            "--output-dir", str(destination), "--gpu", str(gpu)]))
    output.mkdir(parents=True, exist_ok=False)
    (output / "query_plan.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    for index, job in enumerate(jobs, 1):
        log = output / f"{job['bits']}bit.log"
        print(f"{index}/{len(jobs)} Query {job['bits']}bit SAM; log: {log}", flush=True)
        run_process(job["command"], log)
    return query_report(root)


def query_report(root):
    from scripts.evaluate_nabirds_sam_query import validate_run
    plan = check_plan(root)
    registered = {row["bits"]: Path(row["run_dir"]).resolve() for row in training_report(root)["results"]}
    jobs = read_json(root / "query" / "query_plan.json")
    if sorted(job["bits"] for job in jobs) != sorted(plan["bits"]):
        raise ValueError("Query plan does not contain every registered bit length")
    rows, deltas, historical = [], {}, {}
    for job in jobs:
        if (Path(job["run_dir"]).resolve() != registered[job["bits"]]
                or Path(job["destination"]).resolve() != (root / "query" / f"{job['bits']}bit").resolve()):
            raise ValueError("Query plan differs from the registered training/output directories")
        config, selected, _, receipt = validate_run(Path(job["run_dir"]))
        report = read_json(Path(job["destination"]) / "metrics.json")
        if (report.get("query_evaluated") is not True or report["selected_epoch"] != selected["selected_epoch"]
                or config["code_length"] != job["bits"] or config["sam_rho"] != .05
                or report["counts"] != {key: plan["counts"][key] for key in ("query", "database")}
                or any(report.get(k) != v for k, v in receipt.items())
                or report["dataset_content_sha256"] != plan["dataset_content_sha256"]
                or report["split_hash"] != plan["split_hash"]
                or report["evaluation_source_sha256"] != plan["source_sha256"]
                or report["retrieval_outputs_sha256"] != file_sha256(Path(job["destination"]) / "retrieval_outputs.pt")):
            raise ValueError("Query result identity differs from the registered model/data")
        base = next(item for item in plan["items"] if item["bits"] == job["bits"])["historical"]
        values = report["query_metrics"]
        if any(not math.isfinite(values[k]) or not 0 <= values[k] <= 1 for k in METRICS):
            raise ValueError("Invalid query score")
        historical[str(job["bits"])] = base
        deltas[str(job["bits"])] = {key: values[key] - base["query_metrics"][key] for key in METRICS}
        rows.append(dict(bits=job["bits"], selected_epoch=report["selected_epoch"], **values))
    result = dict(dataset="NABirds-B", seed=60, rho=.05, query_evaluated=True,
                  counts={key: plan["counts"][key] for key in ("query", "database")}, results=rows,
                  historical_references=historical, sam_minus_historical_nfd=deltas,
                  note="Only SAM is newly trained/evaluated; original NABirds-B NFD query results are reused.")
    (root / "query" / "query_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (root / "query" / "query_report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--report")
    modes.add_argument("--query")
    modes.add_argument("--query-report")
    parser.add_argument("--baseline-root", default=str(REPO / "save_nabirds_b_i2_nfd"))
    parser.add_argument("--nfd-config", action="append", default=[])
    parser.add_argument("--bits", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--output-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.report or args.query or args.query_report:
            root = Path(args.report or args.query or args.query_report).resolve()
            report = training_report(root) if args.report else query_report(root) if args.query_report else evaluate_queries(root, args.gpu)
            print(json.dumps(report, indent=2), flush=True)
            return
        if (not args.output_root or not args.bits or len(args.bits) != len(set(args.bits))
                or any(bit not in (16, 32, 64) for bit in args.bits) or args.dry_run and args.smoke_test):
            raise ValueError("Supply a new --output-root and unique --bits 16/32/64; choose dry run or smoke test")
        plan = make_plan(args)
        for item in plan["items"]:
            print(f"{item['bits']}bit original NFD: {item['reference_config']}", flush=True)
            print(shlex.join(item["command"]), flush=True)
        print(json.dumps({key: plan[key] for key in ("counts", "split_hash", "dataset_content_sha256", "note")}, indent=2), flush=True)
        if args.dry_run:
            print("Preflight passed; no image loaders iterated, no teacher trained, no query evaluated.", flush=True)
            return
        if not plan["cuda_available"]:
            raise ValueError("CUDA is unavailable")
        if args.smoke_test:
            for item in plan["items"]:
                print(json.dumps(smoke_test(item["command"]), indent=2), flush=True)
            return
        root = Path(args.output_root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        (root / "sam_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        for index, item in enumerate(plan["items"], 1):
            if (source_sha256() != plan["source_sha256"] or file_sha256(__file__) != plan["launcher_sha256"]
                    or file_sha256(item["reference_config"]) != item["reference_config_sha256"]
                    or file_sha256(plan["initialization_file"]) != plan["initialization_sha256"]):
                raise ValueError("Source, reference or ImageNet initialization changed during the experiment")
            log = root / f"{item['bits']}bit.log"
            print(f"{index}/{len(plan['items'])} Starting {item['bits']}bit SAM; log: {log}", flush=True)
            run_process(item["command"], log)
        print(json.dumps(training_report(root), indent=2), flush=True)
        print("Requested SAM training completed; final query is a separate --query operation.", flush=True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"NABirds-B SAM failed: {error}\n")


if __name__ == "__main__":
    main()
