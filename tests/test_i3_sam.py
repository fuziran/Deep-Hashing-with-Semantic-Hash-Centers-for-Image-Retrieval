"""Synthetic SAM correctness, pairing, frozen teacher and validation-only checks."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.sam_nfd import VERSION, sam_nfd_step
from utils.neighbor_feature_distill import NeighborhoodFeatureDistillation
from tests.test_b2nfd_feature_distill import H, S, P, TinyNet, Teacher
from scripts import run_i3_sam_screen as launch


def make_args(path=".", rho=.05):
    return argparse.Namespace(variant="b2nfd", device=torch.device("cpu"), code_length=4,
                              num_classes=3, lr=7e-5, epoch=3, test_map=1, lambd=1e-4,
                              nfd_weight=.01, nfd_warmup=1, nfd_ramp_end=2, skip_query=True,
                              info="synthetic", run_dir=str(path), sam_rho=rho, sam_version=VERSION)


class StepTests(unittest.TestCase):
    def fixture(self):
        from train import CSQLoss
        torch.manual_seed(60)
        args = make_args()
        net = TinyNet(4)
        module = NeighborhoodFeatureDistillation(Teacher(), S, P, topk=2)
        criterion = CSQLoss(args, 4, H, S)
        images = torch.randn(6, 4)
        labels = F.one_hot(torch.arange(6) % 3, 3).float()
        return args, net, module, criterion, images, labels, torch.arange(6)

    def test_matches_manual_sam_gradient_and_restores_before_optimizer(self):
        args, net, module, criterion, x, y, ids = self.fixture()
        manual = copy.deepcopy(net)
        loss = criterion(manual(x), y, ids, args)
        loss.backward()
        norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in manual.parameters()]))
        originals = [p.detach().clone() for p in manual.parameters()]
        with torch.no_grad():
            for p in manual.parameters():
                p.add_(p.grad * args.sam_rho / (norm + 1e-12))
        manual.zero_grad(set_to_none=True)
        criterion(manual(x), y, ids, args).backward()
        expected = [w - .01 * p.grad for w, p in zip(originals, manual.parameters())]
        optimizer = torch.optim.SGD(net.parameters(), lr=.01)
        sam_nfd_step(net, optimizer, criterion, module, x, y, ids, args, 0.)
        for p, value in zip(net.parameters(), expected):
            torch.testing.assert_close(p, value, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(criterion.components["sam_perturbation_norm"], .05, places=6)
        self.assertEqual(net.bn.num_batches_tracked.item(), 1)
        self.assertEqual(module.teacher.calls, 0)

    def test_same_input_randomness_bn_once_and_teacher_once(self):
        args, net, module, criterion, x, y, ids = self.fixture()
        class StochasticNet(TinyNet):
            def __init__(self):
                super().__init__(4)
                self.dropout = nn.Dropout(.25)
                self.masks = []
            def forward_with_features(self, images):
                mask = self.dropout(torch.ones_like(images))
                self.masks.append(mask.detach().clone())
                return super().forward_with_features(images * mask)
        net = StochasticNet()
        ordinary = copy.deepcopy(net)
        torch.manual_seed(802)
        ordinary.forward_with_features(x)
        expected_rng = torch.get_rng_state().clone()
        expected_bn = {key: value.clone() for key, value in ordinary.bn.named_buffers()}
        torch.manual_seed(802)
        sam_nfd_step(net, torch.optim.RMSprop(net.parameters(), lr=args.lr),
                     criterion, module, x, y, ids, args, .01)
        self.assertTrue(torch.equal(net.masks[0], net.masks[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for key, value in net.bn.named_buffers():
            torch.testing.assert_close(value, expected_bn[key], rtol=0, atol=0)
        self.assertEqual(net.bn.momentum, .1)
        self.assertEqual(module.teacher.calls, 1)
        self.assertFalse(module.teacher.training)
        self.assertTrue(all(p.grad is None for p in module.teacher.parameters()))

    def test_failed_second_pass_restores_weights_bn_and_skips_update(self):
        args, net, module, criterion, x, y, ids = self.fixture()
        initial = [p.detach().clone() for p in net.parameters()]
        original = criterion.forward
        calls = []
        def failing(*values):
            calls.append(1)
            if len(calls) == 2:
                raise FloatingPointError("synthetic second pass")
            return original(*values)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
        with patch.object(criterion, "forward", side_effect=failing), \
                patch.object(optimizer, "step") as update, \
                self.assertRaisesRegex(FloatingPointError, "second pass"):
            sam_nfd_step(net, optimizer, criterion, module, x, y, ids, args, .01)
        update.assert_not_called()
        for p, value in zip(net.parameters(), initial):
            self.assertTrue(torch.equal(p, value))
            self.assertIsNone(p.grad)
        self.assertEqual(net.bn.num_batches_tracked.item(), 1)
        self.assertEqual(net.bn.momentum, .1)


class IntegrationTests(unittest.TestCase):
    def tiny(self, path, rho=None):
        from train import train_val
        path.mkdir()
        torch.manual_seed(60)
        args = make_args(path, rho or 0.)
        if rho is None:
            del args.sam_rho, args.sam_version
        nets = []
        def factory(bits):
            net = TinyNet(bits)
            nets.append(net)
            return net
        args.net = factory
        module = NeighborhoodFeatureDistillation(Teacher(), S, P, topk=2)
        loader = DataLoader(TensorDataset(torch.linspace(-1, 1, 48).reshape(12, 4),
                           F.one_hot(torch.arange(12) % 3, 3).float(), torch.arange(12)),
                           batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(60))
        val, query = object(), object()
        def evaluate(selected, *unused):
            self.assertIs(selected, val)
            return ({"mAP@ALL": .5, "mAP@100": .6, "mAP@1000": .55}, None, None, None, None, None)
        with patch("train._evaluate", side_effect=evaluate), patch("train._save_curves"), patch("builtins.print"):
            train_val(args, H, loader, val, query, None, 12, S, nfd=module)
        metrics = json.loads((path / "metrics.json").read_text())
        self.assertIs(metrics["query_evaluated"], False)
        return nets[0], module, json.loads((path / "training_history.json").read_text())

    def test_zero_radius_exact_legacy_reduction_and_full_sam_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, _, oh = self.tiny(root / "legacy")
            control, teacher, ch = self.tiny(root / "control", 0.)
            sam, sam_teacher, sh = self.tiny(root / "sam", .05)
            for key, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, control.state_dict()[key]), key)
            self.assertEqual([r["loss"] for r in oh], [r["loss"] for r in ch])
            self.assertEqual(control.calls, 12)
            self.assertEqual(sam.calls, 24)
            self.assertEqual(control.bn.num_batches_tracked.item(), 12)
            self.assertEqual(sam.bn.num_batches_tracked.item(), 12)
            self.assertEqual(teacher.teacher.calls, 8)
            self.assertEqual(sam_teacher.teacher.calls, 8)
            self.assertEqual([r["train_order_sha256"] for r in ch], [r["train_order_sha256"] for r in sh])
            self.assertEqual([r["first_batch_images_sha256"] for r in ch], [r["first_batch_images_sha256"] for r in sh])
            starts = [json.loads((root / name / "training_start.json").read_text()) for name in ("control", "sam")]
            self.assertEqual(starts[0]["initial_state_sha256"], starts[1]["initial_state_sha256"])
            self.assertGreater(sh[-1]["sam_perturbation_norm"], 0)
            self.assertIn("sampled_nfd_clean_ratio", sh[-1])

    def test_cli_guards_and_distinct_run_identity(self):
        import run
        from tests.test_b2nfd_feature_distill import EntryTests
        ids = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            root.mkdir(exist_ok=True)
            args, argv = EntryTests().fixture(root)
            for extra in (("--sam-rho", "nan"), ("--sam-rho", "-1"),
                          ("--variant", "b1", "--sam-rho", ".05")):
                with patch("sys.argv", argv + list(extra)), patch("sys.stderr"), self.assertRaises(SystemExit):
                    run.load_config()
            for rho in (0., .05):
                with patch("sys.argv", argv + ["--sam-rho", str(rho)]):
                    selected = run.load_config()
                selected.split_hash = "split"
                selected.output_dir = str(root / str(rho))
                loader = argparse.Namespace(generator=torch.Generator())
                from tests.test_b2nfd_feature_distill import CacheTeacher
                with patch("run.load_config", return_value=selected), patch("network.ClassifyNet", CacheTeacher), \
                        patch("run.load_data", return_value=(loader, None, None, None, None, 0, 0, 0, 1)), \
                        patch("run.train_val"), patch("builtins.print"):
                    run.main()
                ids.append(selected.run_id)
            self.assertNotEqual(*ids)

    def test_real_smoke_path_performs_one_update_without_held_out_iteration(self):
        args = make_args()
        args.seed = 60
        nets = []
        def factory(bits):
            net = TinyNet(bits)
            nets.append(net)
            return net
        args.net = factory
        teacher = NeighborhoodFeatureDistillation(Teacher(), S, P, topk=2)
        training = DataLoader(TensorDataset(torch.randn(6, 4),
                              F.one_hot(torch.arange(6) % 3, 3).float(), torch.arange(6)),
                              batch_size=3, generator=torch.Generator())
        class Forbidden:
            def __iter__(self):
                raise AssertionError("Smoke must never iterate held-out data")
        loaders = (training,) + tuple(Forbidden() for _ in range(8))
        with patch("run.load_config", return_value=args), patch("data.data_loader.load_data", return_value=loaders), \
                patch("utils.frozen_b1.load_frozen_b1", return_value=(S, H)), \
                patch("utils.neighbor_feature_distill.load_nfd", return_value=teacher), \
                patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                patch("torch.cuda.max_memory_allocated", return_value=2**30), \
                patch("torch.cuda.max_memory_reserved", return_value=2**30):
            report = launch.smoke_test(["python", "run.py"])
        self.assertTrue(report["smoke_passed"])
        self.assertFalse(report["query_evaluated"])
        self.assertFalse(report["validation_evaluated"])
        self.assertEqual(report["optimizer_updates"], 1)
        self.assertEqual(nets[0].calls, 2)
        self.assertEqual(nets[0].bn.num_batches_tracked.item(), 1)
        self.assertEqual(teacher.teacher.calls, 1)


class LauncherTests(unittest.TestCase):
    def fixture(self, root):
        from tests.test_i3_launcher import LauncherTests as Existing
        path, config = Existing().fixture(root / "base")
        config.update(num_classes=100, nfd_topk=5, split_hash="split", dataset_content_sha256="data",
                      nfd_teacher_sha256="teacher", nfd_basis_sha256="basis", center_content_sha256="centers")
        path.write_text(json.dumps(config))
        return path, config

    def test_paired_commands_inherit_all_settings_and_only_change_radius_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, config = self.fixture(root)
            arms = launch.build_plan(path, config, root / "out")
            self.assertEqual([a["rho"] for a in arms], [0., .05])
            first, second = [a["command"] for a in arms]
            for field in launch.SHARED + launch.ARTIFACTS:
                flag = "--" + field.replace("_", "-")
                self.assertEqual(first[first.index(flag) + 1], second[second.index(flag) + 1])
            self.assertIn("--skip-query", first)
            self.assertNotIn("best_model_state.pt", " ".join(first + second))
            for key, value in (("epoch", 20), ("sam_rho", .05), ("nfd_alpha", 0.), ("seed", 61)):
                with self.assertRaises(ValueError):
                    launch.build_plan(path, {**config, key: value}, root / "out")

    def test_dry_run_has_no_training_or_output_and_existing_root_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, _ = self.fixture(root)
            out = root / "out"
            argv = ["screen", "--nfd-config", str(path), "--output-root", str(out), "--dry-run"]
            with patch("sys.argv", argv), patch.object(launch, "preflight", return_value={"cuda_available": False}), \
                    patch.object(launch, "reference_validation", return_value={key: .5 for key in launch.METRICS}), \
                    patch("subprocess.run") as process, patch("builtins.print"):
                launch.main()
            process.assert_not_called()
            self.assertFalse(out.exists())
            out.mkdir()
            with patch("sys.argv", argv), patch.object(launch, "preflight") as check, \
                    patch("sys.stderr"), self.assertRaises(SystemExit):
                launch.main()
            check.assert_not_called()

    def test_smoke_dispatches_only_one_disposable_model_and_leaves_output_unused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, _ = self.fixture(root)
            out = root / "out"
            argv = ["screen", "--nfd-config", str(path), "--output-root", str(out), "--smoke-test"]
            with patch("sys.argv", argv), patch.object(launch, "preflight", return_value={"cuda_available": True}), \
                    patch.object(launch, "reference_validation", return_value={key: .5 for key in launch.METRICS}), \
                    patch.object(launch, "smoke_test", return_value={"smoke_passed": True}) as smoke, \
                    patch("subprocess.run") as process, patch("builtins.print"):
                launch.main()
            self.assertEqual(smoke.call_count, 1)
            command = smoke.call_args.args[0]
            self.assertEqual(command[command.index("--sam-rho") + 1], "0.05")
            process.assert_not_called()
            self.assertFalse(out.exists())

    def test_preflight_loads_but_never_iterates_held_out_or_training_loaders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, config = self.fixture(root)
            arms = launch.build_plan(path, config, root / "out")
            checkpoint = root / "hub" / "checkpoints" / "resnet34-b627a593.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic initialization")
            class Forbidden:
                def __iter__(self):
                    raise AssertionError("Preflight must not iterate a dataset")
            def load_data(args):
                return tuple(Forbidden() for _ in range(9))
            with patch("data.data_loader.load_data", side_effect=load_data), \
                    patch("utils.frozen_b1.load_frozen_b1"), \
                    patch("utils.neighbor_feature_distill.load_nfd"), \
                    patch("torch.hub.get_dir", return_value=str(root / "hub")):
                result = launch.preflight(path, config, arms)
            self.assertFalse(result["loaders_iterated"])
            self.assertEqual(result["split_hash"], "split")

    def test_summary_budget_pairing_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, config = self.fixture(root)
            arms = launch.build_plan(path, config, root)
            (root / "launch_plan.json").write_text(json.dumps({"arms": arms, "source_sha256": "source"}))
            configs = {}
            for arm in arms:
                run = Path(arm["output"]) / "runs" / "synthetic"
                run.mkdir(parents=True)
                history = [dict(epoch=e, seconds=1., optimizer_updates=2,
                                student_training_passes=4 if arm["rho"] else 2,
                                teacher_training_passes=2 if e > 20 else 0,
                                train_order_sha256=str(e), first_batch_images_sha256="image")
                           for e in range(1, 301)]
                (run / "training_history.json").write_text(json.dumps(history))
                (run / "training_start.json").write_text(json.dumps({"initial_state_sha256": "initial", "batches_per_epoch": 2}))
                configs[run] = {**config, "sam_rho": arm["rho"], "source_sha256": "source", "skip_query": True}
            def validate(run):
                score = .51 if "sam" in run.parts else .5
                return configs[run], {"query_evaluated": False, "selected_epoch": 300}, \
                    {key: score for key in launch.METRICS}, {}
            with patch("scripts.evaluate_query.validate_run", side_effect=validate):
                report = launch.summarize(root)
                self.assertTrue(report["development_gate_passed"])
                broken = root / "sam" / "runs" / "synthetic" / "training_history.json"
                history = json.loads(broken.read_text())
                history[1]["train_order_sha256"] = "different"
                broken.write_text(json.dumps(history))
                with self.assertRaisesRegex(ValueError, "input mismatch"):
                    launch.summarize(root)


if __name__ == "__main__":
    unittest.main()
