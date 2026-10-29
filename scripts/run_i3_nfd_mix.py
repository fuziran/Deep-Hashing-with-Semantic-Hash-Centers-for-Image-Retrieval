"""Launch one paired 32-bit NFD plus center-Mixup pilot; validation only."""
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


def build_command(config_path, options):
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    if (config.get("variant") != "b2nfd" or config.get("dataset") != "cifar-100-new-seg"
            or config.get("seed") != 60 or config.get("code_length") != 32
            or config.get("nfd_alpha", 0) <= 0 or config.get("nfd_weight", 0) <= 0
            or config.get("paired_stage_seeds") is not True
            or config.get("topK") != [-1, 100, 1000]):
        raise ValueError("Expected paired 32-bit seed-60 B1+NFD reference")
    for key in SHARED + ARTIFACTS + ("root", "non_deterministic"):
        if key not in config:
            raise ValueError(f"NFD reference missing {key}")
    root = Path(options.data_root or config["root"])
    if not root.is_absolute():
        root = ROOT / root
    if not root.is_dir():
        raise FileNotFoundError(root)
    output = Path(options.output_root).resolve() / options.mode
    if output.exists():
        raise FileExistsError(output)
    command = [sys.executable, str(ROOT / "run.py"), "--variant", "b3nmix",
               "--paired-stage-seeds", "--skip-query"]
    for key in SHARED:
        command.extend(("--" + key.replace("_", "-"), str(config[key])))
    for key in ARTIFACTS:
        command.extend(("--" + key.replace("_", "-"),
                        str(resolve_file(config[key], config_path))))
    command.extend(("--root", str(root.resolve()), "--topK", "-1", "100", "1000",
                    "--gpu", str(options.gpu), "--mix-mode", options.mode,
                    "--mix-weight", str(options.weight), "--mix-budget", str(options.budget),
                    "--mix-warmup", str(options.warmup),
                    "--mix-ramp-end", str(options.ramp_end),
                    "--output-dir", str(output)))
    if config["non_deterministic"]:
        command.append("--non-deterministic")
    return command, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("uniform_budgeted", "semantic"), required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight", type=float, default=.1)
    parser.add_argument("--budget", type=float, default=2.)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--ramp-end", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if (not math.isfinite(options.weight) or options.weight <= 0
            or not math.isfinite(options.budget) or options.budget <= 0
            or not 0 <= options.warmup < options.ramp_end):
        parser.error("Invalid mixup weight, budget or schedule")
    try:
        command, output = build_command(options.nfd_config, options)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if options.dry_run:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / f"{options.mode}_launch_plan.json").write_text(
        json.dumps({"command": command, "nfd_config": str(Path(options.nfd_config).resolve())},
                   indent=2), encoding="utf-8")
    with (output.parent / f"{options.mode}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    runs = [p for p in (output / "runs").iterdir() if p.is_dir()]
    if len(runs) != 1:
        raise RuntimeError("Expected exactly one completed run")
    metrics = read_config(runs[0] / "metrics.json")
    history = json.loads((runs[0] / "training_history.json").read_text(encoding="utf-8"))
    if metrics.get("query_evaluated") is not False or len(history) != 300:
        raise RuntimeError("Pilot did not finish 300 epochs with final query skipped")
    selected = next(row for row in history if row["epoch"] == metrics["selected_epoch"])
    record = dict(mode=options.mode, run_dir=str(runs[0]),
                  selected_epoch=metrics["selected_epoch"], query_evaluated=False,
                  **selected["validation"])
    with (output.parent / f"{options.mode}_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
