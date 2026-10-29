"""Focused checks for the teacher gate, NFD controls, and pilot launcher."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.diagnose_rgnfd_gate import inspect, summarize, analyze, compare_validation
from scripts.run_i3_npa import SHARED, ARTIFACTS
from scripts.run_i3_rgnfd import build_command
from utils.neighbor_feature_distill import NeighborhoodFeatureDistillation, RELIABLE_VERSION


SIM = torch.tensor([[1., .8, .1], [.8, 1., .3], [.1, .3, 1.]])
PROTO = torch.eye(3, 4)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.classify_layer = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.classify_layer.weight.copy_(torch.tensor([
                [1., 0., 0., 1.], [0., 1., 0., 0.], [0., 0., 1., 0.]]))

    def forward_features(self, images):
        return images + .1


class ReliableNFDTests(unittest.TestCase):
    def test_gate_changes_only_untrusted_sample_and_keeps_teacher_frozen(self):
        module = NeighborhoodFeatureDistillation(Teacher(), SIM, PROTO, alpha=.75, topk=2)
        images = torch.tensor([[2., 0., 0., 2.], [0., 2., 0., 0.]])
        targets = torch.tensor([0, 1])
        student = torch.tensor([[1., .4, .2, .7], [.2, 1., .9, .3]], requires_grad=True)
        teacher_features = module.teacher.forward_features(images)
        expected_gated = module.feature_loss(student, teacher_features, targets,
                                             torch.tensor([1., 0.]))
        expected_fixed = module.feature_loss(student, teacher_features, targets)
        expected_mean = module.feature_loss(student, teacher_features, targets,
                                            torch.full((2,), .5))
        module.gate_mode = "gated"
        actual_gated = module(student, images, targets)
        torch.testing.assert_close(actual_gated, expected_gated)
        self.assertEqual(module.components["rgnfd_raw_gate_fraction"], .5)
        module.gate_mode = "mean"
        torch.testing.assert_close(module(student, images, targets), expected_mean)
        module.gate_mode = "fixed"
        self.assertTrue(torch.equal(module(student, images, targets), expected_fixed))
        actual_gated.backward()
        self.assertGreater(float(student.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in module.teacher.parameters()))

    def test_gate_audit_covers_exact_ids(self):
        class TinyDataset(Dataset):
            indices = [3, 5]
            labels = [0, 0, 0, 0, 0, 1]
            images = torch.tensor([[2., 0., 0., 2.], [0., 2., 0., 0.]])

            def __len__(self):
                return 2

            def __getitem__(self, index):
                image_id = self.indices[index]
                return (self.images[index],
                        torch.nn.functional.one_hot(torch.tensor(self.labels[image_id]), 3).float(),
                        torch.tensor(image_id))

        rows = inspect(DataLoader(TinyDataset(), batch_size=2), Teacher(),
                       torch.device("cpu"), "tiny")
        self.assertEqual([r["image_id"] for r in rows], [3, 5])
        self.assertEqual([r["gate"] for r in rows], [True, False])
        self.assertEqual(summarize(rows)["gate_fraction"], .5)

    def test_gate_audit_does_not_iterate_final_query_or_database(self):
        class TinyDataset(Dataset):
            indices = [3, 5]
            labels = [0, 0, 0, 0, 0, 1]
            images = torch.tensor([[2., 0., 0., 2.], [0., 2., 0., 0.]])

            def __len__(self):
                return 2

            def __getitem__(self, index):
                image_id = self.indices[index]
                return (self.images[index],
                        torch.nn.functional.one_hot(torch.tensor(self.labels[image_id]), 3).float(),
                        torch.tensor(image_id))

        class Forbidden:
            def __iter__(self):
                raise AssertionError("Final query or database was iterated")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            config_path = root / "config.json"
            config_path.write_text(json.dumps(dict(
                variant="b2nfd", nfd_alpha=.75, seed=60, code_length=32,
                paired_stage_seeds=True, root=str(raw), split_hash="split",
                teacher_cache="synthetic-teacher.pt")))
            loader = DataLoader(TinyDataset(), batch_size=2)

            def fake_load(args):
                args.split_hash = "split"
                args.dataset_content_sha256 = "pixels"
                return loader, loader, loader, Forbidden(), Forbidden(), 2, 2, 2, 0, 0

            def fake_nfd(args):
                args.nfd_teacher_sha256 = "teacher"
                args.nfd_basis_sha256 = "basis"
                return argparse.Namespace(teacher=Teacher())

            options = argparse.Namespace(nfd_config=str(config_path), audit_root=None,
                                         output_dir=str(root / "output"), data_root=None,
                                         gpu=0, cpu=True, num_workers=0, dry_run=False)
            with patch("scripts.diagnose_rgnfd_gate.load_data", side_effect=fake_load), \
                 patch("scripts.diagnose_rgnfd_gate.load_nfd", side_effect=fake_nfd):
                report = analyze(options)
            self.assertEqual(report["training"]["gate_fraction"], .5)
            self.assertTrue((root / "output" / "gate_diagnostic.json").is_file())

    def test_validation_comparison_uses_recomputed_data_hash_for_legacy_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "nfd_run"
            ordinary = root / "fd_run"
            audit_root = root / "audit"
            for run, alpha in ((reference, .75), (ordinary, 0)):
                run.mkdir()
                (run / "config.json").write_text(json.dumps(dict(
                    variant="b2nfd", nfd_alpha=alpha)), encoding="utf-8")
            for name, run, values in (("nfd", reference, (.6, .2)),
                                      ("ordinary_fd", ordinary, (.5, .3))):
                audit = audit_root / name
                audit.mkdir(parents=True)
                (audit / "diagnostic.json").write_text(json.dumps(dict(
                    run_dir=str(run), split_hash="split", dataset_content_sha256="pixels",
                    code_length=32, seed=60)), encoding="utf-8")
                with (audit / "per_query_validation.csv").open("w", newline="") as stream:
                    stream.write("validation_id,binary_hamming_mAP@ALL\n")
                    stream.write(f"3,{values[0]}\n5,{values[1]}\n")
            result = compare_validation(
                [dict(image_id=3, gate=True), dict(image_id=5, gate=False)],
                audit_root,
                dict(split_hash="split", code_length=32, seed=60, nfd_alpha=.75),
                reference, "pixels")
            self.assertAlmostEqual(
                result["trusted"]["nfd_minus_ordinary_binary_mAP_ALL"], .1)
            self.assertAlmostEqual(
                result["untrusted"]["nfd_minus_ordinary_binary_mAP_ALL"], -.1)

    def test_launcher_inherits_reference_and_never_asks_for_query(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            files = {}
            for key in ARTIFACTS:
                path = root / (key + ".pt")
                path.touch()
                files[key] = str(path)
            config = {key: 1 for key in SHARED}
            config.update(files)
            config.update(variant="b2nfd", dataset="cifar-100-new-seg", seed=60,
                          code_length=32, nfd_alpha=.75, nfd_weight=.01,
                          paired_stage_seeds=True, topK=[-1, 100, 1000],
                          root=str(raw), non_deterministic=False,
                          split_hash="split", dataset_content_sha256="pixels",
                          nfd_teacher_sha256="teacher", nfd_basis_sha256="basis")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            report_path = root / "report.json"
            report_path.write_text(json.dumps(dict(
                version=RELIABLE_VERSION, training_gate_has_variation=True,
                reference_run=str(root),
                split_hash="split", dataset_content_sha256="pixels",
                teacher_sha256="teacher", projection_sha256="basis")))
            options = argparse.Namespace(data_root=None, output_root=str(root / "output"),
                                         mode="gated", gpu=0)
            command = build_command(config_path, report_path, options)
            self.assertIn("b3rnd", command)
            self.assertIn("--skip-query", command)
            self.assertNotIn("--evaluate-query", command)
            self.assertEqual(command[command.index("--nfd-alpha") + 1], "0.75")
            del config["dataset_content_sha256"]
            config_path.write_text(json.dumps(config))
            self.assertIn("b3rnd", build_command(config_path, report_path, options))
            report = json.loads(report_path.read_text())
            report["projection_sha256"] = "wrong"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Gate report"):
                build_command(config_path, report_path, options)

    def test_fixed_control_preserves_original_nfd_student_updates(self):
        from train import train_val

        class TinyStudent(nn.Module):
            def __init__(self, bits):
                super().__init__()
                self.feature = nn.Linear(4, 4)
                self.hash_layer = nn.Linear(4, bits)

            def forward_with_features(self, x):
                features = self.feature(x).tanh() + .5
                return self.hash_layer(features), features

            def forward(self, x):
                return self.forward_with_features(x)[0]

        x = torch.linspace(-1, 1, 48).reshape(12, 4)
        labels = torch.nn.functional.one_hot(torch.arange(12) % 3, 3).float()
        centers = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.],
                                [-1., -1., -1., -1.]])

        def run_one(root, variant, gate_mode):
            torch.manual_seed(60)
            module = NeighborhoodFeatureDistillation(Teacher(), SIM, PROTO,
                                                    alpha=.75, topk=2)
            module.gate_mode = gate_mode
            loader = DataLoader(torch.utils.data.TensorDataset(x, labels, torch.arange(12)),
                                batch_size=3, shuffle=True,
                                generator=torch.Generator().manual_seed(60))
            args = argparse.Namespace(variant=variant, device=torch.device("cpu"),
                                      code_length=4, num_classes=3, lr=7e-5,
                                      epoch=2, test_map=1, lambd=1e-4,
                                      nfd_weight=.01, nfd_warmup=0, nfd_ramp_end=1,
                                      skip_query=True, info="synthetic", run_dir=str(root),
                                      net=TinyStudent)
            root.mkdir()
            with patch("train._evaluate", return_value=(
                    {"mAP@ALL": .5, "mAP@100": .6, "mAP@1000": .55},
                    None, None, None, None, None)), patch("train._save_curves"), \
                 patch("builtins.print"):
                train_val(args, centers, loader, object(), object(), None, 12,
                          SIM, nfd=module)
            return (torch.load(root / "best_model_state.pt", weights_only=True),
                    json.loads((root / "training_history.json").read_text()))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, base_history = run_one(root / "base", "b2nfd", None)
            fixed, fixed_history = run_one(root / "fixed", "b3rnd", "fixed")
            for key in baseline:
                self.assertTrue(torch.equal(baseline[key], fixed[key]), key)
            self.assertEqual([row["loss"] for row in base_history],
                             [row["loss"] for row in fixed_history])

    def test_new_variant_requires_a_bound_gate_report_and_skip_query(self):
        from run import load_config
        from tests.test_b2nfd_feature_distill import EntryTests

        with tempfile.TemporaryDirectory() as folder:
            args, old_argv = EntryTests().fixture(folder)
            report_path = Path(folder) / "gate.json"
            report_path.write_text(json.dumps(dict(
                version=RELIABLE_VERSION, training_gate_has_variation=True,
                training={"gate_fraction": .7}, split_hash="split",
                dataset_content_sha256="pixels", teacher_sha256=args.classifier_sha256,
                projection_sha256="basis")))
            argv = ["run.py", "--variant", "b3rnd"] + old_argv[3:] + [
                "--rgnfd-gate-diagnostic", str(report_path), "--rgnfd-mode", "mean"]
            with patch("sys.argv", argv):
                parsed = load_config()
            self.assertEqual(parsed.variant, "b3rnd")
            self.assertEqual(parsed.rgnfd_mode, "mean")
            self.assertEqual(parsed.rgnfd_gate_training_fraction, .7)
            with patch("sys.argv", argv[:-4]), patch("sys.stderr"), self.assertRaises(SystemExit):
                load_config()


if __name__ == "__main__":
    unittest.main()
