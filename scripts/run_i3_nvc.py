"""Run one paired 32-bit NVC pilot/control; validation only, never final query."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_i3_npa import SHARED, ARTIFACTS, read_config, resolve_file
from utils.experiment import file_sha256
from utils.neighborhood_view_consistency import VERSION


def build_command(config_path, report_path, options):
    config_path, report_path = Path(config_path).resolve(), Path(report_path).resolve()
    config, report = read_config(config_path), read_config(report_path)
    if (config.get("variant") != "b2nfd" or config.get("dataset") != "cifar-100-new-seg"
            or config.get("seed") != 60 or config.get("code_length") != 32
            or config.get("nfd_alpha", 0) <= 0 or config.get("nfd_weight", 0) <= 0
            or config.get("paired_stage_seeds") is not True
            or config.get("topK") != [-1, 100, 1000]):
        raise ValueError("Expected paired 32-bit seed-60 NFD reference")
    if (report.get("version") != VERSION or report.get("passed") is not True
            or Path(report.get("reference_run", "")).resolve() != config_path.parent
            or report.get("reference_config_sha256") != file_sha256(config_path)
            or report.get("reference_checkpoint_sha256") != file_sha256(
                config_path.parent / "best_model_state.pt")
            or report.get("nfd_alpha") != config.get("nfd_alpha")
            or report.get("nfd_weight") != config.get("nfd_weight")
            or report.get("split_hash") != config.get("split_hash")
            or not report.get("dataset_content_sha256")
            or (config.get("dataset_content_sha256") is not None
                and report["dataset_content_sha256"] != config["dataset_content_sha256"])
            or report.get("teacher_sha256") != config.get("nfd_teacher_sha256")
            or report.get("projection_sha256") != config.get("nfd_basis_sha256")):
        raise ValueError("Preflight report does not match the frozen NFD run or did not pass")
    for key in SHARED + ARTIFACTS + ("root", "non_deterministic"):
        if key not in config:
            raise ValueError(f"NFD reference missing {key}")
    data_root = Path(options.data_root or config["root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    command = [sys.executable, str(ROOT / "run.py"), "--variant", "b3nvc",
               "--paired-stage-seeds", "--skip-query"]
    for key in SHARED:
        command.extend(("--" + key.replace("_", "-"), str(config[key])))
    for key in ARTIFACTS:
        command.extend(("--" + key.replace("_", "-"), str(resolve_file(config[key], config_path))))
    command.extend(("--root", str(data_root.resolve()), "--topK", "-1", "100", "1000",
                    "--gpu", str(options.gpu), "--nvc-mode", options.mode,
                    "--nvc-preflight", str(report_path), "--nvc-weight", str(options.weight),
                    "--nvc-strong-weight", str(options.strong_weight),
                    "--nvc-temperature", str(report["temperature"]),
                    "--nvc-margin", str(report["margin"]),
                    "--nvc-warmup", str(options.warmup),
                    "--nvc-ramp-end", str(options.ramp_end),
                    "--output-dir", str(Path(options.output_root).resolve() / options.mode)))
    if config["non_deterministic"]:
        command.append("--non-deterministic")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("augment", "plain", "semantic"), required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight", type=float, default=.02)
    parser.add_argument("--strong-weight", type=float, default=.2)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--ramp-end", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if (not math.isfinite(options.weight) or options.weight <= 0
            or not math.isfinite(options.strong_weight)
            or not 0 < options.strong_weight <= 1
            or not 0 <= options.warmup < options.ramp_end):
        parser.error("Invalid NVC pilot weights or schedule")
    output_root = Path(options.output_root).resolve()
    if (output_root / options.mode).exists():
        parser.error(f"Output already exists: {output_root / options.mode}")
    try:
        command = build_command(options.nfd_config, options.preflight, options)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if options.dry_run:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{options.mode}_launch_plan.json").write_text(json.dumps(dict(
        mode=options.mode, command=command,
        nfd_config=str(Path(options.nfd_config).resolve()),
        preflight=str(Path(options.preflight).resolve())), indent=2), encoding="utf-8")
    with (output_root / f"{options.mode}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    runs = [p for p in (output_root / options.mode / "runs").iterdir() if p.is_dir()]
    if len(runs) != 1:
        raise RuntimeError("Expected exactly one completed run")
    metrics = read_config(runs[0] / "metrics.json")
    history = json.loads((runs[0] / "training_history.json").read_text(encoding="utf-8"))
    if metrics.get("query_evaluated") is not False or len(history) != 300:
        raise RuntimeError("Pilot did not finish 300 epochs with final query skipped")
    selected = next(r for r in history if r["epoch"] == metrics["selected_epoch"])
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
