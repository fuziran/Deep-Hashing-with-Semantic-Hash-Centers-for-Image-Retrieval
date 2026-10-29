"""Launch one paired 32-bit reliability-gated NFD pilot or its matched controls.

Requires a frozen seed-60 B1+NFD reference and a completed training-only gate
feasibility report. Only validation selects a checkpoint; query is always skipped.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_i3_npa import SHARED, ARTIFACTS, read_config, resolve_file
from utils.neighbor_feature_distill import RELIABLE_VERSION


def build_command(config_path, report_path, options):
    config_path = Path(config_path).resolve()
    report_path = Path(report_path).resolve()
    config = read_config(config_path)
    report = read_config(report_path)
    if (config.get("variant") != "b2nfd" or config.get("dataset") != "cifar-100-new-seg"
            or config.get("seed") != 60 or config.get("code_length") != 32
            or config.get("nfd_alpha", 0) <= 0 or config.get("nfd_weight", 0) <= 0
            or config.get("paired_stage_seeds") is not True
            or config.get("topK") != [-1, 100, 1000]):
        raise ValueError("Expected the paired 32-bit seed-60 NFD reference")
    if (report.get("version") != RELIABLE_VERSION
            or report.get("training_gate_has_variation") is not True
            or Path(report.get("reference_run", "")).resolve() != config_path.parent
            or report.get("split_hash") != config.get("split_hash")
            or report.get("dataset_content_sha256") != config.get("dataset_content_sha256")
            or report.get("teacher_sha256") != config.get("nfd_teacher_sha256")
            or report.get("projection_sha256") != config.get("nfd_basis_sha256")):
        raise ValueError("Gate report does not match the frozen NFD run or lacks variation")
    for key in SHARED + ARTIFACTS + ("root", "topK", "non_deterministic"):
        if key not in config:
            raise ValueError(f"Reference missing {key}")
    data_root = Path(options.data_root or config["root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    command = [sys.executable, str(ROOT / "run.py"), "--variant", "b3rnd",
               "--paired-stage-seeds", "--skip-query"]
    for key in SHARED:
        command.extend(("--" + key.replace("_", "-"), str(config[key])))
    for key in ARTIFACTS:
        command.extend(("--" + key.replace("_", "-"), str(resolve_file(config[key], config_path))))
    command.extend(("--root", str(data_root.resolve()), "--topK", "-1", "100", "1000",
                    "--gpu", str(options.gpu), "--rgnfd-mode", options.mode,
                    "--rgnfd-gate-diagnostic", str(report_path),
                    "--output-dir", str(Path(options.output_root).resolve() / options.mode)))
    if config["non_deterministic"]:
        command.append("--non-deterministic")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-config", required=True)
    parser.add_argument("--gate-diagnostic", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("gated", "mean", "fixed"), required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    output_root = Path(options.output_root).resolve()
    output = output_root / options.mode
    if output.exists():
        parser.error(f"Output already exists: {output}")
    try:
        command = build_command(options.nfd_config, options.gate_diagnostic, options)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if options.dry_run:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    plan = dict(mode=options.mode, command=command,
                nfd_config=str(Path(options.nfd_config).resolve()),
                gate_diagnostic=str(Path(options.gate_diagnostic).resolve()))
    (output_root / f"{options.mode}_launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    with (output_root / f"{options.mode}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    runs = [p for p in (output / "runs").iterdir() if p.is_dir()]
    if len(runs) != 1:
        raise RuntimeError(f"Expected exactly one saved run under {output}")
    metrics = read_config(runs[0] / "metrics.json")
    history = json.loads((runs[0] / "training_history.json").read_text(encoding="utf-8"))
    if metrics.get("query_evaluated") is not False or len(history) != 300:
        raise RuntimeError("Pilot did not finish all epochs with query skipped")
    selected = next(row for row in history if row["epoch"] == metrics["selected_epoch"])
    record = dict(mode=options.mode, run_dir=str(runs[0]),
                  selected_epoch=metrics["selected_epoch"], query_evaluated=False,
                  **selected["validation"])
    with (output_root / f"{options.mode}_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
