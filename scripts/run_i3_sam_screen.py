"""Fixed seed-60 NFD versus ordinary SAM control at 16/32/64 bits; validation only."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_i3_npa import SHARED, ARTIFACTS, read_config, resolve_file

RHO = .05
METRICS = ("mAP@ALL", "mAP@100", "mAP@1000")


def build_plan(path, config, root, gpu=0, data_root=None, sam_only=False):
    if (config.get("variant") != "b2nfd" or config.get("paired_stage_seeds") is not True
            or config.get("dataset") != "cifar-100-new-seg"
            or config.get("code_length") not in (16, 32, 64) or config.get("seed") != 60
            or config.get("epoch") != 300 or config.get("num_classes") != 100
            or config.get("non_deterministic") is not False
            or config.get("nfd_alpha") != .75 or config.get("nfd_weight") != .01
            or config.get("nfd_topk") != 5 or config.get("nfd_warmup") != 20
            or config.get("nfd_ramp_end") != 40 or config.get("sam_rho", 0) != 0):
        raise ValueError("Expected an original deterministic 16/32/64-bit seed-60, 300-epoch B1+NFD reference")
    for key in SHARED + ARTIFACTS + ("root", "topK", "split_hash"):
        if key not in config:
            raise ValueError(f"Missing reference field: {key}")
    if config["topK"] != [-1, 100, 1000]:
        raise ValueError("Unexpected reference cutoffs")
    data = Path(data_root or config["root"])
    if not data.is_absolute():
        data = REPO / data
    if not data.is_dir():
        raise ValueError(f"Data root does not exist: {data}")
    common = [sys.executable, str(REPO / "run.py"), "--variant", "b2nfd",
              "--paired-stage-seeds", "--skip-query"]
    for key in SHARED:
        common += ["--" + key.replace("_", "-"), str(config[key])]
    for key in ARTIFACTS:
        common += ["--" + key.replace("_", "-"), str(resolve_file(config[key], path))]
    common += ["--root", str(data.resolve()), "--gpu", str(gpu), "--topK", "-1", "100", "1000"]
    modes = (("sam", RHO),) if sam_only else (("control", 0.), ("sam", RHO))
    return [dict(mode=mode, rho=rho, output=str(root / mode),
                 command=common + ["--sam-rho", str(rho), "--output-dir", str(root / mode)])
            for mode, rho in modes]


def reference_data_identity(path, config, reference_audit=None, dataset_anchor_run=None):
    """Bind a legacy manifest to an existing checkpoint-specific dataset audit.

    Never fill the expected fingerprint from the data currently being checked.
    Historical manifests are left unchanged; no scores select an audit.
    """
    from utils.experiment import file_sha256
    expected = config.get("dataset_content_sha256")
    if dataset_anchor_run:
        if reference_audit:
            raise ValueError("Choose either --reference-audit or --dataset-anchor-run")
        from scripts.evaluate_query import validate_run
        anchor = Path(dataset_anchor_run).resolve()
        anchor_config, _, _, receipt = validate_run(anchor)
        if (anchor_config.get("variant") != "b2nfd" or anchor_config.get("code_length") != 32
                or anchor_config.get("sam_rho", 0) != 0):
            raise ValueError("Dataset anchor must be the completed ordinary 32-bit NFD control")
        for field in ("dataset", "seed", "num_classes", "val_per_class", "split_hash"):
            if anchor_config.get(field) != config.get(field):
                raise ValueError(f"Dataset anchor protocol mismatch: {field}")
        digest = anchor_config["dataset_content_sha256"]
        if expected and expected != digest:
            raise ValueError("Reference config and dataset anchor fingerprints disagree")
        return dict(expected_dataset_content_sha256=digest,
                    dataset_identity_source="completed-paired-32bit-control",
                    dataset_anchor_run=str(anchor), dataset_anchor_receipt=receipt)
    audit_path = Path(reference_audit).resolve() if reference_audit else None
    if expected and audit_path is None:
        return dict(expected_dataset_content_sha256=expected,
                    dataset_identity_source="reference-config")
    if audit_path is None:
        pointer = REPO / "logs" / "i3_bottleneck_latest.path"
        if pointer.is_file():
            folder = Path(pointer.read_text(encoding="utf-8").strip())
            if not folder.is_absolute():
                folder = REPO / folder
            audit_path = (folder / "nfd" / "diagnostic.json").resolve()
    if audit_path is None or not audit_path.is_file():
        raise ValueError("Legacy NFD config has no dataset_content_sha256. Supply --reference-audit "
                         "with the existing NFD bottleneck diagnostic.json, or restore "
                         "logs/i3_bottleneck_latest.path; do not edit the original config.")
    audit = read_config(audit_path)
    checkpoint = path.parent / "best_model_state.pt"
    for field, value in (("variant", "b2nfd"), ("code_length", config["code_length"]),
                         ("seed", config["seed"]), ("split_hash", config["split_hash"]),
                         ("training_source_sha256", config["source_sha256"]),
                         ("checkpoint_sha256", file_sha256(checkpoint))):
        if not value or audit.get(field) != value:
            raise ValueError(f"Reference audit identity mismatch: {field}")
    digest = audit.get("dataset_content_sha256")
    if not isinstance(digest, str) or not digest:
        raise ValueError("Reference audit is missing dataset_content_sha256")
    if expected and digest != expected:
        raise ValueError("Reference config and audit dataset fingerprints disagree")
    return dict(expected_dataset_content_sha256=digest,
                dataset_identity_source="checkpoint-specific-bottleneck-audit",
                reference_audit=str(audit_path), reference_audit_sha256=file_sha256(audit_path),
                reference_checkpoint_sha256=audit["checkpoint_sha256"])


def preflight(path, config, plan, data_root=None, reference_audit=None, dataset_anchor_run=None):
    """Validate raw data and frozen parent artifacts, without iterating any loader."""
    import torch
    from data.data_loader import load_data
    from utils.experiment import file_sha256, seed_everything
    from utils.frozen_b1 import load_frozen_b1
    from utils.neighbor_feature_distill import load_nfd
    identity = reference_data_identity(path, config, reference_audit, dataset_anchor_run)
    args = argparse.Namespace(**config)
    args.variant, args.device = "b2nfd", torch.device("cpu")
    args.root = plan[0]["command"][plan[0]["command"].index("--root") + 1]
    for key in ARTIFACTS:
        setattr(args, key, str(resolve_file(config[key], path)))
    with tempfile.TemporaryDirectory(prefix="shc_sam_preflight_") as folder:
        args.output_dir = folder
        seed_everything(config["seed"])
        loaders = load_data(args)
        del loaders  # No training, validation, final-query or database iteration.
        if (args.split_hash != config["split_hash"]
                or args.dataset_content_sha256 != identity["expected_dataset_content_sha256"]):
            raise ValueError("Reference dataset pixels/labels or split mismatch")
        load_frozen_b1(args)
        module = load_nfd(args)
        for field in ("nfd_teacher_sha256", "nfd_basis_sha256", "center_content_sha256"):
            if getattr(args, field) != config[field]:
                raise ValueError(f"Reference artifact identity mismatch: {field}")
        del module
    # Require an existing ImageNet initialization; this screen never downloads it.
    weights = REPO / "models_ckpt" / "resnet34-b627a593.pth"
    if not weights.is_file():
        weights = Path(torch.hub.get_dir()) / "checkpoints" / weights.name
    if not weights.is_file():
        raise FileNotFoundError("Existing resnet34-b627a593.pth is required in models_ckpt or torch hub cache")
    return dict(split_hash=args.split_hash, dataset_content_sha256=args.dataset_content_sha256,
                **identity,
                initialization_file=str(weights.resolve()), initialization_sha256=file_sha256(weights),
                teacher_sha256=args.nfd_teacher_sha256, basis_sha256=args.nfd_basis_sha256,
                cuda_available=torch.cuda.is_available(), loaders_iterated=False)


def reference_validation(path):
    metrics = read_config(path.parent / "metrics.json")
    history = json.loads((path.parent / "training_history.json").read_text(encoding="utf-8"))
    if len(history) != 300 or [row.get("epoch") for row in history] != list(range(1, 301)):
        raise ValueError("Original NFD reference has incomplete training history")
    selected = max((row for row in history if "validation" in row),
                   key=lambda row: row["validation"]["mAP@ALL"])
    if selected["epoch"] != metrics["selected_epoch"]:
        raise ValueError("Original NFD reference selection disagrees with validation history")
    return {key: selected["validation"][key] for key in METRICS}


def smoke_test(command):
    """One disposable real GPU SAM batch at full NFD weight; no evaluation."""
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
    with tempfile.TemporaryDirectory(prefix="shc_sam_smoke_") as folder:
        args.output_dir = folder
        seed_everything(args.seed)
        loaders = load_data(args)
        training = loaders[0]
        del loaders
        similarity, centers = load_frozen_b1(args)
        nfd = load_nfd(args)
        seed_everything(args.seed)
        training.generator.manual_seed(args.seed)
        net = args.net(args.code_length).to(args.device).train()
        if centers.shape != (args.num_classes, args.code_length):
            centers = centers.T
        criterion = CSQLoss(args, args.code_length, centers.float(), similarity)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-5)
        images, labels, indices = next(iter(training))
        images, labels = images.to(args.device), labels.to(args.device)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        value, _ = sam_nfd_step(net, optimizer, criterion, nfd, images, labels, indices,
                                args, args.nfd_weight, sample_gradients=True)
        torch.cuda.synchronize(args.device)
        return dict(smoke_passed=True, batch_size=len(images), full_nfd_weight=args.nfd_weight,
                    clean_objective=value, seconds=time.monotonic() - started,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device) / 2**30,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved(args.device) / 2**30,
                    sam_perturbation_norm=criterion.components["sam_perturbation_norm"],
                    query_evaluated=False, validation_evaluated=False, optimizer_updates=1,
                    note="Disposable smoke model; formal paired runs start fresh and use the complete budget.")


def save_summary(root, report):
    (root / "screen_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["results"][0]))
        writer.writeheader()
        writer.writerows(report["results"])
    return report


def summarize(root):
    from scripts.evaluate_query import validate_run
    plan = read_config(root / "launch_plan.json")
    sam_only = plan.get("sam_only", False)
    if [arm["mode"] for arm in plan["arms"]] != (["sam"] if sam_only else ["control", "sam"]):
        raise ValueError("Planned training arms do not match the comparison mode")
    rows, runs = [], {}
    for arm in plan["arms"]:
        folders = [p for p in (Path(arm["output"]) / "runs").iterdir() if p.is_dir()]
        if len(folders) != 1:
            raise ValueError(f"Expected exactly one completed {arm['mode']} run")
        run = folders[0]
        config, metrics, validation, _ = validate_run(run)
        history = json.loads((run / "training_history.json").read_text(encoding="utf-8"))
        start = read_config(run / "training_start.json")
        if (metrics.get("query_evaluated") is not False or not config.get("skip_query")
                or config.get("sam_rho") != arm["rho"] or len(history) != 300
                or config.get("source_sha256") != plan["source_sha256"]):
            raise ValueError(f"Unexpected run protocol: {run}")
        if not all(row.get("optimizer_updates") == start["batches_per_epoch"]
                   and row.get("student_training_passes") == start["batches_per_epoch"] * (2 if arm["rho"] else 1)
                   and row.get("teacher_training_passes") == (start["batches_per_epoch"] if row["epoch"] > 20 else 0)
                   for row in history):
            raise ValueError("Training pass/update budget mismatch")
        runs[arm["mode"]] = (config, history, start)
        rows.append(dict(mode=arm["mode"], code_length=config["code_length"], run_dir=str(run), selected_epoch=metrics["selected_epoch"],
                         completed_epochs=len(history), rho=arm["rho"], query_evaluated=False,
                         epoch_loop_seconds=sum(row["seconds"] for row in history), **validation))
    if sam_only:
        reference = plan["reference_validation"]
        deltas = {key: rows[0][key] - reference[key] for key in METRICS}
        return save_summary(root, dict(code_length=runs["sam"][0]["code_length"], seed=60,
            results=rows, comparison="historical-reference", query_evaluated=False,
            historical_reference_validation=reference, sam_minus_historical_reference=deltas,
            sam_minus_control=None, paired_initialization=None, paired_epoch_orders=None,
            paired_first_batch_image_checks=None,
            development_gate_passed=deltas["mAP@ALL"] >= .005 and all(deltas[k] >= 0 for k in METRICS),
            note="SAM-only training; the original NFD reference is reused, not retrained. No newly paired control or query evaluation is implied."))
    control, sam = runs["control"], runs["sam"]
    if control[2]["initial_state_sha256"] != sam[2]["initial_state_sha256"]:
        raise ValueError("Paired initialization mismatch")
    for field in SHARED + ("split_hash", "dataset_content_sha256", "source_sha256",
                           "center_content_sha256", "nfd_teacher_sha256", "nfd_basis_sha256"):
        if control[0][field] != sam[0][field]:
            raise ValueError(f"Paired protocol mismatch: {field}")
    for first, second in zip(control[1], sam[1]):
        for field in ("train_order_sha256", "first_batch_images_sha256"):
            if not first.get(field) or first[field] != second.get(field):
                raise ValueError(f"Paired training input mismatch at epoch {first['epoch']}: {field}")
    deltas = {key: rows[1][key] - rows[0][key] for key in METRICS}
    reference = plan.get("reference_validation")
    historical_delta = {key: rows[1][key] - reference[key] for key in METRICS} if reference else None
    report = dict(code_length=control[0]["code_length"], seed=control[0]["seed"],
                  results=rows, paired_initialization=True, paired_epoch_orders=True,
                  paired_first_batch_image_checks=True, query_evaluated=False,
                  historical_reference_validation=reference,
                  control_minus_historical_reference={key: rows[0][key] - reference[key] for key in METRICS} if reference else None,
                  sam_minus_historical_reference=historical_delta,
                  sam_minus_control=deltas,
                  development_gate_passed=(deltas["mAP@ALL"] >= .005 and all(deltas[k] >= 0 for k in METRICS)
                                           and (historical_delta is None or
                                                historical_delta["mAP@ALL"] >= .005 and
                                                all(historical_delta[k] >= 0 for k in METRICS))),
                  note="Existing-method direction screen, not innovation 3. Epochs and updates match; SAM uses twice the student passes. Same selection validation, no independent confirmation or causal flatness claim.")
    return save_summary(root, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfd-config")
    parser.add_argument("--output-root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--reference-audit", help="Existing checkpoint-specific NFD diagnostic.json for a legacy config")
    parser.add_argument("--dataset-anchor-run", help="Completed 32-bit ordinary NFD control with a verified data fingerprint")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="One disposable GPU SAM batch with full NFD, no evaluation")
    parser.add_argument("--sam-only", action="store_true", help="Train only SAM, reusing the historical NFD reference")
    parser.add_argument("--report", help="Verify and summarize an already completed paired screen")
    options = parser.parse_args()
    try:
        if options.report:
            print(json.dumps(summarize(Path(options.report).resolve()), indent=2), flush=True)
            return
        if not options.nfd_config or not options.output_root:
            raise ValueError("Supply --nfd-config and --output-root")
        path, root = Path(options.nfd_config).resolve(), Path(options.output_root).resolve()
        if root.exists():
            raise FileExistsError(f"Choose a new output root: {root}")
        config = read_config(path)
        arms = build_plan(path, config, root, options.gpu, options.data_root, sam_only=options.sam_only)
        checks = preflight(path, config, arms, options.data_root, options.reference_audit, options.dataset_anchor_run)
        from utils.experiment import file_sha256, source_sha256
        plan = dict(reference_config=str(path), reference_config_sha256=file_sha256(path),
                    source_sha256=source_sha256(), checks=checks, arms=arms, sam_only=options.sam_only,
                    reference_validation=reference_validation(path))
        for arm in arms:
            print(f"{arm['mode']}: {shlex.join(arm['command'])}", flush=True)
        print(json.dumps(checks, indent=2), flush=True)
        if options.dry_run and options.smoke_test:
            raise ValueError("Choose either --dry-run or --smoke-test")
        if options.dry_run:
            print("Preflight passed; no network trained, no loader iterated.", flush=True)
            return
        if not checks["cuda_available"]:
            raise ValueError("CUDA is unavailable; real screen requires the original SHC GPU environment")
        if options.smoke_test:
            print(json.dumps(smoke_test(next(arm for arm in arms if arm["mode"] == "sam")["command"]), indent=2), flush=True)
            return
        root.mkdir(parents=True, exist_ok=False)
        (root / "launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        completed = []
        for index, arm in enumerate(arms, 1):
            if (source_sha256() != plan["source_sha256"] or file_sha256(path) != plan["reference_config_sha256"]
                    or file_sha256(checks["initialization_file"]) != checks["initialization_sha256"]):
                raise ValueError("Source, reference config or initialization changed after preflight")
            print(f"{index}/{len(arms)} Starting {arm['mode']}; detailed log: {root / (arm['mode'] + '.log')}", flush=True)
            started = time.monotonic()
            with (root / (arm["mode"] + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(arm["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                               check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            completed.append(dict(mode=arm["mode"], wall_seconds=time.monotonic() - started))
            (root / "completed_arms.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
        print(json.dumps(summarize(root), indent=2), flush=True)
        print("Requested training runs completed. Report: " + str(root / "screen_report.json"), flush=True)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SAM screen failed: {error}\n")


if __name__ == "__main__":
    main()
