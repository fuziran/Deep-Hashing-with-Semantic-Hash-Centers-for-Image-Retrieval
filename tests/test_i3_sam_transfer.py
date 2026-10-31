"""Cross-bit configuration inheritance, legacy data anchors and separated query."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_i3_sam_screen as single
from scripts import run_i3_sam_transfer as transfer
from tests import test_i3_sam as fixtures
from utils.experiment import file_sha256, source_sha256


class TransferTests(unittest.TestCase):
    def test_actual_tiny_nfd_sam_training_at_both_new_lengths(self):
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from train import train_val
        from tests.test_b2nfd_feature_distill import H, S, P, TinyNet, Teacher
        from utils.neighbor_feature_distill import NeighborhoodFeatureDistillation
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for bit in (16, 64):
                starts = []
                for mode, rho in (("control", 0.), ("sam", .05)):
                    destination = root / f"{bit}bit" / mode
                    destination.mkdir(parents=True)
                    args = fixtures.make_args(destination, rho)
                    args.code_length, args.net = bit, TinyNet
                    torch.manual_seed(60)
                    nfd = NeighborhoodFeatureDistillation(Teacher(), S, P, topk=2)
                    loader = DataLoader(TensorDataset(torch.linspace(-1, 1, 48).reshape(12, 4),
                                        F.one_hot(torch.arange(12) % 3, 3).float(), torch.arange(12)),
                                        batch_size=3, generator=torch.Generator().manual_seed(60), shuffle=True)
                    val, query = object(), object()
                    def evaluate(selected, *unused):
                        self.assertIs(selected, val)
                        return ({"mAP@ALL": .5, "mAP@100": .6, "mAP@1000": .55}, None, None, None, None, None)
                    with patch("train._evaluate", side_effect=evaluate), patch("train._save_curves"), patch("builtins.print"):
                        train_val(args, H.repeat(1, bit // 4), loader, val, query, None, 12, S, nfd=nfd)
                    self.assertFalse(json.loads((destination / "metrics.json").read_text())["query_evaluated"])
                    state = torch.load(destination / "best_model_state.pt", weights_only=True)
                    self.assertEqual(tuple(state["hash_layer.weight"].shape), (bit, 4))
                    starts.append(json.loads((destination / "training_start.json").read_text())["initial_state_sha256"])
                    self.assertEqual(nfd.teacher.calls, 8)
                self.assertEqual(starts[0], starts[1])

    def fixture(self, root):
        paths, configs = {}, {}
        for bit in transfer.BITS:
            path, config = fixtures.LauncherTests().fixture(root / str(bit))
            config["code_length"] = bit
            del config["dataset_content_sha256"]
            path.write_text(json.dumps(config))
            paths[bit], configs[bit] = path, config
        options = argparse.Namespace(output_root=str(root / "out"), dataset_anchor_run=str(root / "anchor"),
                                     baseline_root=str(root), nfd_config=[], data_root=None, gpu=0)
        return paths, configs, options

    def checks(self):
        return dict(split_hash="split", dataset_content_sha256="data", initialization_sha256="initial", cuda_available=True)

    def test_each_bit_inherits_its_own_artifacts_and_fixed_rho(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths, configs, options = self.fixture(root)
            with patch.object(single, "preflight", return_value=self.checks()) as preflight, \
                    patch.object(single, "reference_validation", return_value={k: .5 for k in single.METRICS}):
                plan = transfer.make_plan(options)
            self.assertEqual([i["bits"] for i in plan["items"]], [16, 64])
            for item, call in zip(plan["items"], preflight.call_args_list):
                bit = item["bits"]
                self.assertTrue(plan["sam_only"])
                self.assertEqual([arm["mode"] for arm in item["arms"]], ["sam"])
                self.assertIn("--sam-only", item["command"])
                self.assertEqual(item["reference_config"], str(paths[bit]))
                self.assertEqual(call.kwargs["dataset_anchor_run"], root / "anchor")
                for arm in item["arms"]:
                    command = arm["command"]
                    for key in single.SHARED:
                        flag = "--" + key.replace("_", "-")
                        self.assertEqual(command[command.index(flag) + 1], str(configs[bit][key]))
                    self.assertEqual(command[command.index("--center-cache") + 1], configs[bit]["center_cache"])
                    self.assertEqual(command[command.index("--sam-rho") + 1], "0.05" if arm["mode"] == "sam" else "0.0")
                    self.assertIn("--skip-query", command)
            self.assertFalse(Path(options.output_root).exists())

    def test_historical_query_prefers_saved_query_and_labels_table_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths, _, _ = self.fixture(root)
            for bit in (16, 64):
                fallback = transfer.historical_query(paths[bit], bit)
                self.assertEqual(fallback["source_kind"], "user-reported-query-table")
                self.assertEqual(fallback["displayed_decimals"], 6)
                metrics = {key: .55 for key in single.METRICS}
                saved = paths[bit].parent / "metrics.json"
                saved.write_text(json.dumps(dict(query_evaluated=False, query_metrics=None, best_validation_mAP=.99)))
                self.assertEqual(transfer.historical_query(paths[bit], bit)["query_metrics"], transfer.USER_REPORTED_QUERY[bit])
                saved.write_text(json.dumps(dict(query_evaluated=True, query_metrics=metrics)))
                reference = transfer.historical_query(paths[bit], bit)
                self.assertEqual(reference["source_kind"], "saved-query-metrics")
                self.assertEqual(reference["query_metrics"], metrics)
                self.assertEqual(reference["source_sha256"], file_sha256(saved))

    def test_sam_only_summary_needs_no_new_control_and_never_claims_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths, configs, _ = self.fixture(root)
            output = root / "only"
            output.mkdir()
            arms = single.build_plan(paths[16], configs[16], output, sam_only=True)
            (output / "launch_plan.json").write_text(json.dumps(dict(arms=arms, sam_only=True, source_sha256="source",
                reference_validation={key: .5 for key in single.METRICS})))
            run = output / "sam" / "runs" / "selected"
            run.mkdir(parents=True)
            history = [dict(epoch=e, seconds=1., optimizer_updates=2, student_training_passes=4,
                            teacher_training_passes=2 if e > 20 else 0) for e in range(1, 301)]
            (run / "training_history.json").write_text(json.dumps(history))
            (run / "training_start.json").write_text(json.dumps(dict(batches_per_epoch=2)))
            config = {**configs[16], "sam_rho": .05, "skip_query": True, "source_sha256": "source"}
            with patch("scripts.evaluate_query.validate_run", return_value=(config,
                       {"query_evaluated": False, "selected_epoch": 300}, {key: .51 for key in single.METRICS}, {})):
                report = single.summarize(output)
            self.assertEqual(report["comparison"], "historical-reference")
            self.assertIsNone(report["paired_initialization"])
            self.assertIsNone(report["sam_minus_control"])
            self.assertAlmostEqual(report["sam_minus_historical_reference"]["mAP@ALL"], .01)
            self.assertFalse((output / "control").exists())

    def test_ordinary_fd_excluded_and_ambiguous_reference_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths, configs, _ = self.fixture(root)
            other = root / "ordinary_fd" / "config.json"
            other.parent.mkdir()
            other.write_text(json.dumps({**configs[16], "nfd_alpha": 0.}))
            self.assertEqual(set(transfer.references(root, [])), {16, 64})
            other.write_text(json.dumps(configs[16]))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                transfer.references(root, [])
            selected = transfer.references(root, [str(paths[16]), str(paths[64])])
            self.assertEqual(selected[16][0], paths[16])
            with self.assertRaisesRegex(ValueError, "Missing"):
                transfer.references(root, [str(paths[16])])

    def test_legacy_cross_bit_anchor_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths, configs, _ = self.fixture(root)
            anchor = {**configs[16], "code_length": 32, "sam_rho": 0., "dataset_content_sha256": "data"}
            validation = (anchor, {}, {}, {"checkpoint_sha256": "checkpoint"})
            with patch("scripts.evaluate_query.validate_run", return_value=validation):
                for bit in (16, 64):
                    identity = single.reference_data_identity(paths[bit], configs[bit], dataset_anchor_run=root / "anchor")
                    self.assertEqual(identity["expected_dataset_content_sha256"], "data")
                    self.assertEqual(identity["dataset_identity_source"], "completed-paired-32bit-control")
                with self.assertRaisesRegex(ValueError, "seed"):
                    single.reference_data_identity(paths[16], {**configs[16], "seed": 61}, dataset_anchor_run=root / "anchor")
                with self.assertRaisesRegex(ValueError, "fingerprints"):
                    single.reference_data_identity(paths[16], {**configs[16], "dataset_content_sha256": "changed"}, dataset_anchor_run=root / "anchor")
                with self.assertRaisesRegex(ValueError, "either"):
                    single.reference_data_identity(paths[16], configs[16], reference_audit="audit", dataset_anchor_run=root / "anchor")
            with patch("scripts.evaluate_query.validate_run", return_value=({**anchor, "sam_rho": .05}, {}, {}, {})), \
                    self.assertRaisesRegex(ValueError, "ordinary 32-bit"):
                single.reference_data_identity(paths[16], configs[16], dataset_anchor_run=root / "anchor")

    def test_dry_run_checks_all_and_training_never_dispatches_query(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, options = self.fixture(root)
            argv = ["transfer", "--baseline-root", str(root), "--dataset-anchor-run", options.dataset_anchor_run,
                    "--output-root", options.output_root]
            with patch.object(single, "preflight", return_value=self.checks()), \
                    patch.object(single, "reference_validation", return_value={k: .5 for k in single.METRICS}), \
                    patch.object(transfer, "evaluate_queries") as query, \
                    patch.object(transfer, "run_process") as run, patch("builtins.print"):
                with patch("sys.argv", argv + ["--dry-run"]):
                    transfer.main()
                self.assertFalse(Path(options.output_root).exists())
                run.assert_not_called()
                with patch("sys.argv", argv + ["--smoke-test"]), \
                        patch.object(single, "smoke_test", return_value={"smoke_passed": True}) as smoke:
                    transfer.main()
                self.assertEqual(smoke.call_count, 2)
                self.assertFalse(Path(options.output_root).exists())
                with patch("sys.argv", argv), patch.object(transfer, "training_report", return_value={"query_evaluated": False}):
                    transfer.main()
                self.assertEqual(run.call_count, 2)
                self.assertTrue(all("--sam-only" in call.args[0] for call in run.call_args_list))
                self.assertEqual([call.args[0][call.args[0].index("--nfd-config") + 1] for call in run.call_args_list],
                                 [str(root / "16" / "base" / "config.json"), str(root / "64" / "base" / "config.json")])
                query.assert_not_called()

    def query_fixture(self, root):
        plan = dict(version=transfer.VERSION, source_sha256=source_sha256(), seed=60, rho=.05,
                    items=[dict(bits=bit, checks=self.checks()) for bit in (16, 64)])
        (root / "transfer_plan.json").write_text(json.dumps(plan))
        rows, configs = [], {}
        for bit in (16, 64):
            for mode in ("control", "sam"):
                run = root / f"{bit}bit" / mode / "runs" / "selected"
                rows.append(dict(code_length=bit, mode=mode, run_dir=str(run)))
                configs[run] = dict(code_length=bit, seed=60, sam_rho=.05 if mode == "sam" else 0.,
                                    split_hash="split", dataset_content_sha256="data")
        return dict(results=rows), configs

    def test_four_queries_preflight_before_forward_and_receipts_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training, configs = self.query_fixture(root)
            events = []
            def dry(options):
                self.assertTrue(options.dry_run)
                events.append("dry")
                return {"split_hash": "split", "dataset_content_sha256": "data"}
            def run(command, log):
                self.assertEqual(events[:4], ["dry"] * 4)
                events.append("query")
                selected = Path(command[command.index("--run-dir") + 1])
                destination = Path(command[command.index("--output-dir") + 1])
                destination.mkdir(parents=True)
                payload = destination / "retrieval_outputs.pt"
                payload.write_bytes(b"synthetic outputs")
                config = configs[selected]
                report = dict(**config, query_evaluated=True, selected_epoch=285,
                              counts={"query": 5000, "database": 45000}, checkpoint_sha256="checkpoint",
                              evaluation_source_sha256=source_sha256(), retrieval_outputs_sha256=file_sha256(payload),
                              query_metrics={key: .51 if config["sam_rho"] else .5 for key in single.METRICS})
                (destination / "metrics.json").write_text(json.dumps(report))
            def validate(run):
                return configs[run], {"selected_epoch": 285}, {}, {"checkpoint_sha256": "checkpoint"}
            with patch.object(transfer, "training_report", return_value=training), \
                    patch("scripts.evaluate_query.evaluate", side_effect=dry), \
                    patch("scripts.evaluate_query.validate_run", side_effect=validate), \
                    patch.object(transfer, "run_process", side_effect=run), patch("builtins.print"):
                report = transfer.evaluate_queries(root, 0)
                self.assertEqual(len(report["results"]), 4)
                self.assertEqual(events, ["dry"] * 4 + ["query"] * 4)
                for bit in ("16", "64"):
                    self.assertAlmostEqual(report["sam_minus_control"][bit]["mAP@ALL"], .01)
                payload = root / "query" / "64bit" / "sam" / "retrieval_outputs.pt"
                payload.write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "artifact identity"):
                    transfer.query_report(root)

    def test_failed_query_preflight_leaves_no_output_or_query_forward(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training, _ = self.query_fixture(root)
            with patch.object(transfer, "training_report", return_value=training), \
                    patch("scripts.evaluate_query.evaluate", side_effect=ValueError("receipt mismatch")), \
                    patch.object(transfer, "run_process") as run, \
                    self.assertRaisesRegex(ValueError, "receipt"):
                transfer.evaluate_queries(root, 0)
            run.assert_not_called()
            self.assertFalse((root / "query").exists())

    def test_sam_only_query_evaluates_two_new_models_and_reuses_old_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training, configs = self.query_fixture(root)
            training["results"] = [row for row in training["results"] if row["mode"] == "sam"]
            plan = json.loads((root / "transfer_plan.json").read_text())
            plan["sam_only"] = True
            for item in plan["items"]:
                item["historical_query"] = dict(query_metrics=transfer.USER_REPORTED_QUERY[item["bits"]],
                    source_kind="user-reported-query-table", displayed_decimals=6)
            (root / "transfer_plan.json").write_text(json.dumps(plan))
            events = []
            def dry(options):
                self.assertTrue(options.dry_run)
                self.assertIn("sam", Path(options.run_dir).parts)
                events.append("dry")
                return {"split_hash": "split", "dataset_content_sha256": "data"}
            def run(command, log):
                self.assertEqual(events[:2], ["dry", "dry"])
                events.append("query")
                selected = Path(command[command.index("--run-dir") + 1])
                self.assertIn("sam", selected.parts)
                destination = Path(command[command.index("--output-dir") + 1])
                destination.mkdir(parents=True)
                payload = destination / "retrieval_outputs.pt"
                payload.write_bytes(b"SAM query outputs")
                config = configs[selected]
                report = dict(**config, query_evaluated=True, selected_epoch=300,
                    counts={"query": 5000, "database": 45000}, checkpoint_sha256="checkpoint",
                    evaluation_source_sha256=source_sha256(), retrieval_outputs_sha256=file_sha256(payload),
                    query_metrics={key: .7 for key in single.METRICS})
                (destination / "metrics.json").write_text(json.dumps(report))
            with patch.object(transfer, "training_report", return_value=training), \
                    patch("scripts.evaluate_query.evaluate", side_effect=dry), \
                    patch("scripts.evaluate_query.validate_run", side_effect=lambda p: (
                        configs[p], {"selected_epoch": 300}, {}, {"checkpoint_sha256": "checkpoint"})), \
                    patch.object(transfer, "run_process", side_effect=run), patch("builtins.print"):
                report = transfer.evaluate_queries(root, 0)
            self.assertEqual(events, ["dry", "dry", "query", "query"])
            self.assertEqual(len(report["results"]), 2)
            self.assertIsNone(report["sam_minus_control"])
            self.assertEqual(report["comparison"], "historical-reference")
            self.assertAlmostEqual(report["sam_minus_reference_query"]["16"]["mAP@ALL"], .7 - .541192)
            self.assertAlmostEqual(report["sam_minus_reference_query"]["64"]["mAP@ALL"], .7 - .598426)
            self.assertFalse((root / "query" / "16bit" / "control").exists())
            self.assertFalse((root / "query" / "64bit" / "control").exists())


if __name__ == "__main__":
    unittest.main()
