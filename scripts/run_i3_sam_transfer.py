"""Fixed 16/64-bit seed-60 transfer of the successful rho=0.05 SAM recipe.

Train four matched arms sequentially. Query evaluation is a separate explicit
operation after every validation-selected model is saved, for all four arms.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_i3_sam_screen as single
from scripts.run_i3_npa import read_config
from utils.experiment import file_sha256, source_sha256

BITS = (16, 64)
VERSION = "i3-nfd-sam-fixed-transfer-v1"


def references(baseline_root, explicit):
    paths = [Path(p).resolve() for p in explicit] if explicit else sorted(Path(baseline_root).resolve().rglob("config.json"))
    selected = {}
    for path in paths:
        config = read_config(path)
        if (config.get("variant") != "b2nfd" or config.get("code_length") not in BITS
                or config.get("dataset") != "cifar-100-new-seg" or config.get("seed") != 60
                or config.get("nfd_alpha") != .75 or config.get("nfd_weight") != .01
                or config.get("sam_rho", 0) != 0):
            if explicit:
                raise ValueError(f"Not an original 16/64-bit NFD reference: {path}")
            continue
        bit = config["code_length"]
        if bit in selected:
            raise ValueError(f"Ambiguous {bit}-bit NFD references; supply two explicit --nfd-config paths")
        selected[bit] = path, config
    if set(selected) != set(BITS):
        raise ValueError(f"Missing original NFD references for {sorted(set(BITS) - set(selected))}; "
                         "check --baseline-root or supply one --nfd-config for each length")
    return selected


def make_plan(options):
    root = Path(options.output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Choose a new transfer output root: {root}")
    anchor = Path(options.dataset_anchor_run).resolve()
    selected = references(options.baseline_root, options.nfd_config)
    items = []
    for bit in BITS:
        path, config = selected[bit]
        output = root / f"{bit}bit"
        arms = single.build_plan(path, config, output, options.gpu, options.data_root)
        checks = single.preflight(path, config, arms, options.data_root, dataset_anchor_run=anchor)
        reference = single.reference_validation(path)
        command = [sys.executable, str(REPO / "scripts" / "run_i3_sam_screen.py"),
                   "--nfd-config", str(path), "--output-root", str(output),
                   "--dataset-anchor-run", str(anchor), "--gpu", str(options.gpu)]
        if options.data_root:
            command += ["--data-root", str(Path(options.data_root).resolve())]
        items.append(dict(bits=bit, reference_config=str(path), reference_config_sha256=file_sha256(path),
                          output=str(output), arms=arms, checks=checks,
                          reference_validation=reference, command=command))
    if any(items[0]["checks"][key] != items[1]["checks"][key]
           for key in ("split_hash", "dataset_content_sha256", "initialization_sha256")):
        raise ValueError("16/64-bit data split, pixels or ImageNet initialization differ")
    return dict(version=VERSION, source_sha256=source_sha256(), seed=60, rho=single.RHO,
                dataset_anchor_run=str(anchor), items=items,
                launcher_sha256=file_sha256(__file__), single_launcher_sha256=file_sha256(single.__file__))


def check_plan(root):
    plan = read_config(root / "transfer_plan.json")
    if (plan.get("version") != VERSION or plan.get("source_sha256") != source_sha256()
            or plan.get("seed") != 60 or plan.get("rho") != .05
            or [item["bits"] for item in plan["items"]] != list(BITS)):
        raise ValueError("Transfer plan identity/source mismatch")
    return plan


def save_report(root, name, report):
    (root / (name + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (root / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["results"][0]))
        writer.writeheader()
        writer.writerows(report["results"])
    return report


def training_report(root):
    plan = check_plan(root)
    results, deltas, gates = [], {}, {}
    for item in plan["items"]:
        report = single.summarize(Path(item["output"]))
        if report["code_length"] != item["bits"] or report["seed"] != 60:
            raise ValueError("Child screen has a different bit length or seed")
        results.extend(report["results"])
        deltas[str(item["bits"])] = report["sam_minus_control"]
        gates[str(item["bits"])] = report["development_gate_passed"]
    return save_report(root, "transfer_report", dict(
        seed=60, rho=.05, query_evaluated=False, paired_checks_passed=True,
        results=results, sam_minus_control=deltas, development_gates=gates,
        note="Fixed transfer: 16/64 bits are both reported regardless of validation gain. Query is separate; ordinary SAM is an existing-method control."))


def run_process(command, log_path):
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def evaluate_queries(root, gpu):
    from scripts.evaluate_query import evaluate
    plan = check_plan(root)
    training = training_report(root)
    output = root / "query"
    if output.exists():
        raise FileExistsError(f"Query output already exists; inspect its logs/results before retrying: {output}")
    jobs = []
    for row in training["results"]:
        bit, mode = row["code_length"], row["mode"]
        destination = output / f"{bit}bit" / mode
        # Check all four selections before iterating any query loader.
        checked = evaluate(argparse.Namespace(run_dir=row["run_dir"], output_dir=str(destination),
                                             data_root=None, gpu=gpu, cpu=False, dry_run=True))
        item = next(i for i in plan["items"] if i["bits"] == bit)
        for field in ("split_hash", "dataset_content_sha256"):
            if checked[field] != item["checks"][field]:
                raise ValueError(f"Query data identity differs from preflight: {field}")
        command = [sys.executable, str(REPO / "scripts" / "evaluate_query.py"),
                   "--run-dir", row["run_dir"], "--output-dir", str(destination), "--gpu", str(gpu)]
        jobs.append(dict(bits=bit, mode=mode, run_dir=row["run_dir"], destination=str(destination), command=command))
    output.mkdir(parents=True, exist_ok=False)
    (output / "query_plan.json").write_text(json.dumps(dict(version=VERSION, jobs=jobs), indent=2), encoding="utf-8")
    for index, job in enumerate(jobs, 1):
        log = output / f"{job['bits']}bit_{job['mode']}.log"
        print(f"{index}/4 Query {job['bits']}bit {job['mode']}; log: {log}", flush=True)
        run_process(job["command"], log)
    return query_report(root)


def query_report(root):
    from scripts.evaluate_query import validate_run
    plan = check_plan(root)
    jobs = read_config(root / "query" / "query_plan.json")["jobs"]
    if {(job["bits"], job["mode"]) for job in jobs} != {(b, m) for b in BITS for m in ("control", "sam")} or len(jobs) != 4:
        raise ValueError("Expected all four fixed query evaluations")
    results, values, identities = [], {}, set()
    for job in jobs:
        config, selected, _, receipt = validate_run(Path(job["run_dir"]))
        report = read_config(Path(job["destination"]) / "metrics.json")
        if (report.get("query_evaluated") is not True or config["seed"] != 60
                or config["code_length"] != job["bits"] or report["code_length"] != job["bits"]
                or config.get("sam_rho") != (.05 if job["mode"] == "sam" else 0.)
                or report["selected_epoch"] != selected["selected_epoch"]
                or report["counts"] != {"query": 5000, "database": 45000}
                or any(report.get(key) != value for key, value in receipt.items())
                or report["split_hash"] != config["split_hash"]
                or report["dataset_content_sha256"] != config["dataset_content_sha256"]):
            raise ValueError("Query report does not match its fixed training selection")
        if (report["evaluation_source_sha256"] != plan["source_sha256"]
                or file_sha256(Path(job["destination"]) / "retrieval_outputs.pt") != report["retrieval_outputs_sha256"]):
            raise ValueError("Query evaluation source or retrieval artifact identity mismatch")
        identities.add((report["split_hash"], report["dataset_content_sha256"]))
        metrics = {key: report["query_metrics"][key] for key in single.METRICS}
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
               for value in metrics.values()):
            raise ValueError("Invalid query metric")
        values[job["bits"], job["mode"]] = metrics
        results.append(dict(code_length=job["bits"], mode=job["mode"], selected_epoch=report["selected_epoch"], **metrics))
    if len(identities) != 1:
        raise ValueError("Query dataset identities differ across transfer runs")
    deltas = {str(bit): {key: values[bit, "sam"][key] - values[bit, "control"][key]
                        for key in single.METRICS} for bit in BITS}
    return save_report(root / "query", "query_report", dict(
        seed=60, rho=.05, query_evaluated=True, counts=dict(query=5000, database=45000),
        results=results, sam_minus_control=deltas,
        note="Single-seed fixed-parameter transfer of existing SAM, not a novelty claim or untouched-test evidence for later tuning."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--report")
    modes.add_argument("--query", help="Evaluate all four completed selections; new ROOT/query only")
    modes.add_argument("--query-report")
    parser.add_argument("--baseline-root", default=str(REPO / "save_nfd_16_64_seed60_ZZdT2Ou0"))
    parser.add_argument("--nfd-config", action="append", default=[])
    parser.add_argument("--dataset-anchor-run")
    parser.add_argument("--output-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    options = parser.parse_args()
    try:
        if options.report or options.query or options.query_report:
            root = Path(options.report or options.query or options.query_report).resolve()
            function = training_report if options.report else query_report if options.query_report else None
            report = function(root) if function else evaluate_queries(root, options.gpu)
            print(json.dumps(report, indent=2), flush=True)
            return
        if not options.output_root or not options.dataset_anchor_run:
            raise ValueError("Supply --output-root and --dataset-anchor-run for transfer training")
        if options.dry_run and options.smoke_test:
            raise ValueError("Choose either --dry-run or --smoke-test")
        plan = make_plan(options)
        for item in plan["items"]:
            print(f"{item['bits']}bit reference: {item['reference_config']}", flush=True)
            print(shlex.join(item["command"]), flush=True)
            print(json.dumps(item["checks"], indent=2), flush=True)
        if options.dry_run:
            print("Both bit lengths passed preflight; no loaders iterated and no training started.", flush=True)
            return
        if not all(item["checks"]["cuda_available"] for item in plan["items"]):
            raise ValueError("CUDA is unavailable")
        if options.smoke_test:
            for item in plan["items"]:
                print(json.dumps(dict(bits=item["bits"], **single.smoke_test(item["arms"][1]["command"])), indent=2), flush=True)
            return
        root = Path(options.output_root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        (root / "transfer_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        for index, item in enumerate(plan["items"], 1):
            if (source_sha256() != plan["source_sha256"] or file_sha256(__file__) != plan["launcher_sha256"]
                    or file_sha256(single.__file__) != plan["single_launcher_sha256"]
                    or file_sha256(item["reference_config"]) != item["reference_config_sha256"]):
                raise ValueError("Code or reference config changed during transfer")
            print(f"{index}/2 Starting {item['bits']}bit control -> sam; log: {root / (str(item['bits']) + 'bit_launcher.log')}", flush=True)
            run_process(item["command"], root / (str(item["bits"]) + "bit_launcher.log"))
        print(json.dumps(training_report(root), indent=2), flush=True)
        print("All four training runs completed. No query has been evaluated by this command.", flush=True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SAM transfer failed: {error}\n")


if __name__ == "__main__":
    main()
