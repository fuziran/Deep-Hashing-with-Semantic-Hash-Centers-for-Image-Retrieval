"""Focused NVC mechanism, data isolation and launch provenance checks."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from data.data_loader import CIFAR100TwoViewDataset
from scripts.diagnose_i3_nvc_preflight import analyze, inspect
from scripts.run_i3_npa import ARTIFACTS, SHARED
from scripts.run_i3_nvc import build_command
from utils.experiment import file_sha256
from utils.neighborhood_view_consistency import (
    VERSION, semantic_weight, consistency_loss)


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classify_layer = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.classify_layer.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))

    def forward_features(self, image):
        return image


class Student(torch.nn.Module):
    def forward(self, image):
        return image[:, :2]


class NVCTests(unittest.TestCase):
    def test_semantic_projection_screens_neighbor_direction_change(self):
        weak = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
        strong = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
        basis = torch.tensor([[[0.], [1.], [0.]]])
        gate, ratio = semantic_weight(weak, strong, basis, torch.tensor([1]),
                                      torch.tensor([0, 0]), .05)
        self.assertGreater(float(ratio[0]), float(ratio[1]))
        self.assertLess(float(gate[0]), float(gate[1]))
        self.assertTrue(((gate >= 0) & (gate <= 1)).all())

    def test_only_at_risk_center_correct_bits_receive_gradient(self):
        weak = torch.tensor([[1., -1.], [1., -1.]])
        strong = torch.tensor([[-.2, -2.], [-.2, -2.]], requires_grad=True)
        center = torch.tensor([[1., -1.]])
        loss, stats = consistency_loss(weak, strong, torch.tensor([0, 0]),
                                       center, torch.tensor([1., 0.]), .5)
        self.assertEqual(stats["nvc_active_bits"], 2)
        loss.backward()
        self.assertLess(float(strong.grad[0, 0]), 0.)
        self.assertEqual(float(strong.grad[0, 1]), 0.)
        self.assertEqual(float(strong.grad[1].abs().sum()), 0.)

    def test_paired_dataset_keeps_exact_training_ids(self):
        import numpy as np
        from torchvision.transforms import ToTensor

        images = np.zeros((6, 4, 4, 3), dtype=np.uint8)
        images[3] = 50
        images[5] = 200
        dataset = CIFAR100TwoViewDataset(images, np.array([0, 0, 0, 1, 0, 1]),
                                         np.array([3, 5]), 2,
                                         transform=ToTensor(), strong_transform=ToTensor())
        self.assertEqual([int(dataset[i][2]) for i in range(2)], [3, 5])
        self.assertEqual(len(dataset[0]), 4)
        torch.testing.assert_close(dataset[1][0], dataset[1][3])

    def test_training_preflight_never_iterates_final_query(self):
        class TinyDataset(Dataset):
            indices = [3, 5]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.tensor([1., 0., 0.]) if index == 0 else torch.tensor([0., 1., 0.])
                strong = torch.tensor([1., .3, 0.]) if index == 0 else torch.tensor([.3, 1., 0.])
                return image, torch.nn.functional.one_hot(torch.tensor(index), 2).float(), \
                    torch.tensor(self.indices[index]), strong

        class Forbidden:
            def __iter__(self):
                raise AssertionError("Held-out loader was iterated")

        loader = DataLoader(TinyDataset(), batch_size=2)
        result = inspect(loader, Student(), argparse.Namespace(
            teacher=Teacher(), basis=torch.tensor([
                [[1.], [0.], [0.]], [[0.], [1.], [0.]]]),
            ranks=torch.tensor([1, 1])), torch.tensor([[1., 1.], [-1., -1.]]),
            torch.device("cpu"), .05, .5)
        self.assertEqual(result["count"], 2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / "run" / "config.json"
            config_path.parent.mkdir()
            (config_path.parent / "best_model_state.pt").touch()
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            config_path.write_text(json.dumps(dict(
                variant="b2nfd", dataset="cifar-100-new-seg", seed=60,
                code_length=32, num_classes=2, nfd_alpha=.75, nfd_weight=.01, paired_stage_seeds=True,
                root=str(raw), split_hash="split", num_workers=0)))

            def fake_load(args):
                args.split_hash = "split"
                args.dataset_content_sha256 = "pixels"
                return loader, Forbidden(), Forbidden(), Forbidden(), Forbidden(), 2, 0, 0, 0

            options = argparse.Namespace(nfd_config=str(config_path),
                                         output_dir=str(root / "output"),
                                         data_root=None, gpu=0, cpu=True,
                                         num_workers=0, temperature=.05,
                                         margin=.5, dry_run=True)
            with patch("scripts.diagnose_i3_nvc_preflight.load_data", side_effect=fake_load):
                report = analyze(options)
            self.assertTrue(report["dry_run"])
            self.assertFalse((root / "output").exists())
            torch.save({}, config_path.parent / "best_model_state.pt")
            options.dry_run = False

            def fake_nfd(args):
                args.nfd_teacher_sha256 = "teacher"
                args.nfd_basis_sha256 = "basis"
                return argparse.Namespace(
                    teacher=Teacher(), basis=torch.tensor([
                        [[1.], [0.], [0.]], [[0.], [1.], [0.]]]),
                    ranks=torch.tensor([1, 1]))

            class Student32(Student):
                def forward(self, image):
                    return super().forward(image).repeat(1, 16)

            with patch("scripts.diagnose_i3_nvc_preflight.load_data", side_effect=fake_load), \
                 patch("scripts.diagnose_i3_nvc_preflight.load_nfd", side_effect=fake_nfd), \
                 patch("scripts.diagnose_i3_nvc_preflight.ResNet", side_effect=lambda *a, **k: Student32()), \
                 patch("utils.frozen_b1.load_frozen_b1", return_value=(
                     None, torch.tensor([[1.] * 32, [-1.] * 32]))):
                report = analyze(options)
            self.assertEqual(report["training"]["count"], 2)
            self.assertTrue((root / "output" / "preflight.json").is_file())

    def test_launcher_accepts_legacy_data_hash_and_forces_skip_query(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            config = {key: 1 for key in SHARED}
            for key in ARTIFACTS:
                path = root / (key + ".pt")
                path.touch()
                config[key] = str(path)
            config.update(variant="b2nfd", dataset="cifar-100-new-seg", seed=60,
                          code_length=32, nfd_alpha=.75, nfd_weight=.01,
                          paired_stage_seeds=True, topK=[-1, 100, 1000],
                          root=str(raw), non_deterministic=False, split_hash="split",
                          nfd_teacher_sha256="teacher", nfd_basis_sha256="basis")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            checkpoint = root / "best_model_state.pt"
            checkpoint.write_bytes(b"checkpoint")
            report_path = root / "preflight.json"
            report_path.write_text(json.dumps(dict(
                version=VERSION, passed=True, reference_run=str(root),
                reference_config_sha256=file_sha256(config_path),
                reference_checkpoint_sha256=file_sha256(checkpoint),
                nfd_alpha=.75, nfd_weight=.01,
                split_hash="split", dataset_content_sha256="pixels",
                teacher_sha256="teacher", projection_sha256="basis",
                temperature=.05, margin=.5)))
            options = argparse.Namespace(data_root=None, output_root=str(root / "output"),
                                         mode="semantic", gpu=0, weight=.02,
                                         strong_weight=.2, warmup=40, ramp_end=60)
            command = build_command(config_path, report_path, options)
            self.assertIn("--skip-query", command)
            self.assertIn("b3nvc", command)
            report = json.loads(report_path.read_text())
            report["projection_sha256"] = "wrong"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Preflight"):
                build_command(config_path, report_path, options)

    def test_training_modes_finish_with_query_skipped(self):
        from train import train_val

        class TinyNet(torch.nn.Module):
            def __init__(self, bits):
                super().__init__()
                self.feature = torch.nn.Linear(3, 3)
                self.hash_layer = torch.nn.Linear(3, bits)

            def forward_with_features(self, image):
                feature = self.feature(image).tanh() + .5
                return self.hash_layer(feature), feature

            def forward(self, image):
                return self.forward_with_features(image)[0]

        class TinyNFD:
            teacher = Teacher()
            basis = torch.tensor([[[1.], [0.], [0.]], [[0.], [1.], [0.]]])
            ranks = torch.tensor([1, 1])

            def __init__(self):
                self.components = {}
                self.last_teacher_features = None

            def __call__(self, features, image, targets):
                self.last_teacher_features = self.teacher.forward_features(image).detach()
                loss = (features - self.last_teacher_features).square().mean()
                self.components = {"loss_nfd": float(loss.detach())}
                return loss

        images = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                               [1., .2, 0.], [.2, 1., 0.]])
        labels = torch.nn.functional.one_hot(torch.tensor([0, 1, 0, 1]), 2).float()
        strong = torch.tensor([[-1., 0., 0.], [0., -1., 0.],
                               [-1., .2, 0.], [.2, -1., 0.]])
        centers = torch.tensor([[1., 1.], [-1., -1.]])
        with tempfile.TemporaryDirectory() as folder:
            for mode in ("augment", "plain", "semantic"):
                torch.manual_seed(60)
                loader = DataLoader(torch.utils.data.TensorDataset(
                    images, labels, torch.arange(4), strong), batch_size=2,
                    shuffle=True, generator=torch.Generator().manual_seed(60))
                run_dir = Path(folder) / mode
                run_dir.mkdir()
                args = argparse.Namespace(
                    variant="b3nvc", device=torch.device("cpu"), code_length=2,
                    num_classes=2, lr=7e-5, epoch=1, test_map=1, lambd=1e-4,
                    nfd_weight=.01, nfd_warmup=0, nfd_ramp_end=1,
                    nvc_mode=mode, nvc_weight=.02, nvc_strong_weight=.2,
                    nvc_warmup=0, nvc_ramp_end=1, nvc_temperature=.05,
                    nvc_margin=.5, skip_query=True, info="tiny",
                    run_dir=str(run_dir), net=TinyNet)
                with patch("train._evaluate", return_value=(
                        {"mAP@ALL": .5, "mAP@100": .6, "mAP@1000": .55},
                        None, None, None, None, None)), patch("train._save_curves"), \
                     patch("builtins.print"):
                    train_val(args, centers, loader, object(), object(), None, 4,
                              nfd=TinyNFD())
                metrics = json.loads((run_dir / "metrics.json").read_text())
                history = json.loads((run_dir / "training_history.json").read_text())
                self.assertFalse(metrics["query_evaluated"])
                self.assertIn("nvc_risk_bit_fraction", history[0])
                self.assertEqual(history[0]["nvc_weight"] > 0, mode != "augment")

    def test_variant_parser_requires_matching_preflight(self):
        from run import load_config
        from tests.test_b2nfd_feature_distill import EntryTests

        with tempfile.TemporaryDirectory() as folder:
            original, argv = EntryTests().fixture(folder)
            report_path = Path(folder) / "preflight.json"
            report_path.write_text(json.dumps(dict(
                version=VERSION, passed=True, nfd_alpha=.75, nfd_weight=.01,
                temperature=.05, margin=.5, split_hash="split",
                dataset_content_sha256="pixels", teacher_sha256="teacher",
                projection_sha256="basis")))
            command = ["run.py", "--variant", "b3nvc"] + argv[3:] + [
                "--nvc-preflight", str(report_path)]
            with patch("sys.argv", command):
                parsed = load_config()
            self.assertEqual(parsed.variant, "b3nvc")
            self.assertEqual(parsed.nvc_mode, "semantic")
            self.assertTrue(parsed.skip_query)
            report = json.loads(report_path.read_text())
            report["temperature"] = .1
            report_path.write_text(json.dumps(report))
            with patch("sys.argv", command), patch("sys.stderr"), self.assertRaises(SystemExit):
                load_config()


if __name__ == "__main__":
    unittest.main()
