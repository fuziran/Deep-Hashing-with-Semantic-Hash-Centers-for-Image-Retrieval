"""Check fusion-signal accounting and the training-only diagnostic protocol."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.diagnose_i3_fused_target_signal import analyze, inspect, target_statistics


class FusedTargetSignalTests(unittest.TestCase):
    def test_identical_views_have_no_extra_target_or_distillation_signal(self):
        logits = torch.tensor([[3., -3.]])
        stats = target_statistics(logits, logits, logits, torch.tensor([[1., -1.]]), 2.)
        self.assertEqual(stats["fusion_changed_bits"].item(), 0)
        self.assertEqual(stats["target_probability_gap"].item(), 0)
        self.assertAlmostEqual(stats["fused_to_strong_kl"].item(), 0., places=7)
        self.assertTrue(torch.equal(stats["single_gradient"], stats["fused_gradient"]))

    def test_repairs_and_damages_are_measured_against_frozen_b1_center(self):
        stats = target_statistics(torch.tensor([[-2., 2.]]),
                                  torch.tensor([[4., -4.]]),
                                  torch.tensor([[-1., -1.]]),
                                  torch.tensor([[1., -1.]]), 2.)
        self.assertEqual(stats["fusion_changed_bits"].item(), 2)
        self.assertEqual(stats["single_damages"].item(), 1)
        self.assertEqual(stats["fused_repairs"].item(), 1)
        self.assertEqual(stats["fused_damages"].item(), 0)

    def test_reported_output_gradient_matches_autograd(self):
        strong = torch.tensor([[.3, -.7]], requires_grad=True)
        stats = target_statistics(torch.tensor([[1., -.2]]),
                                  torch.tensor([[.4, -.9]]), strong,
                                  torch.tensor([[1., -1.]]), 2.)
        stats["fused_to_strong_kl"].mean().backward()
        self.assertTrue(torch.allclose(strong.grad, stats["fused_gradient"], atol=1e-7))

    def test_inspect_does_not_update_parameters(self):
        class TrainingOnly(Dataset):
            indices = [3, 7]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.tensor([[[1., -1.]]]) * (1 if index == 0 else -1)
                label = torch.nn.functional.one_hot(torch.tensor(index), 2).float()
                return image, image * .8, label, torch.tensor(self.indices[index])

        net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 2))
        state = {key: value.clone() for key, value in net.state_dict().items()}
        summary, _, ids, _ = inspect(DataLoader(TrainingOnly(), batch_size=2), net,
                                     torch.tensor([[1., -1.], [-1., 1.]]),
                                     torch.device("cpu"), 2, 2.)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(ids, [3, 7])
        self.assertTrue(all(torch.equal(value, net.state_dict()[key])
                            for key, value in state.items()))

    def test_dry_run_never_iterates_held_out_loaders_or_constructs_model(self):
        class NoIteration:
            def __iter__(self):
                raise AssertionError("Held-out loader was iterated")

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            config = dict(variant="b2nfd", code_length=32, seed=60, nfd_alpha=.75,
                          split_hash="test-split", root=str(raw))
            run = dict(config=config, path=root, checkpoint_sha256="test-model",
                       summary={"selected_epoch": 300})

            def load(args):
                args.split_hash = "test-split"
                args.dataset_content_sha256 = "test-data"
                return (NoIteration(), SimpleNamespace(dataset=[1, 2]),
                        NoIteration(), NoIteration(), NoIteration())

            options = argparse.Namespace(nfd_run=str(root), output_dir=str(root / "report"),
                                         data_root=None, cpu=True, gpu=0, num_workers=0,
                                         temperature=2., dry_run=True)
            with patch("scripts.diagnose_i3_fused_target_signal.checked_run", return_value=run), \
                 patch("scripts.diagnose_i3_fused_target_signal.load_data", side_effect=load), \
                 patch("scripts.diagnose_i3_fused_target_signal.ResNet",
                       side_effect=AssertionError("Model constructed during dry run")):
                report = analyze(options)
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["training_count"], 2)
            self.assertFalse((root / "report").exists())

    def test_full_report_writes_training_rows_without_held_out_iteration(self):
        class NoIteration:
            def __iter__(self):
                raise AssertionError("Held-out loader was iterated")

        class TrainingOnly(Dataset):
            indices = np.array([0, 1])
            images = np.array([[[[255, 0, 0]]], [[[0, 255, 0]]]], dtype=np.uint8)

            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.tensor(self.images[index]).permute(2, 0, 1).float() / 255
                label = torch.nn.functional.one_hot(torch.tensor(index), 2).float()
                return image, label, torch.tensor(index)

        class TinyNet(torch.nn.Module):
            def __init__(self, bits, pretrained=False):
                super().__init__()
                self.layer = torch.nn.Linear(3, bits)

            def forward(self, image):
                return self.layer(image.flatten(1))

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            checkpoint = root / "model.pt"
            torch.save(TinyNet(32).state_dict(), checkpoint)
            config = dict(variant="b2nfd", code_length=32, seed=60, nfd_alpha=.75,
                          split_hash="test-split", root=str(raw), num_workers=0,
                          batch_size=2, num_classes=2)
            run = dict(config=config, path=root, checkpoint=checkpoint,
                       checkpoint_sha256="test-model", summary={"selected_epoch": 300})

            def load(args):
                args.split_hash = "test-split"
                args.dataset_content_sha256 = "test-data"
                strong = lambda image: torch.tensor(np.array(image)).permute(2, 0, 1).float() / 300
                return (SimpleNamespace(dataset=SimpleNamespace(strong_transform=strong)),
                        SimpleNamespace(dataset=TrainingOnly()),
                        NoIteration(), NoIteration(), NoIteration())

            options = argparse.Namespace(nfd_run=str(root), output_dir=str(root / "report"),
                                         data_root=None, cpu=True, gpu=0, num_workers=0,
                                         temperature=2., dry_run=False)
            with patch("scripts.diagnose_i3_fused_target_signal.checked_run", return_value=run), \
                 patch("scripts.diagnose_i3_fused_target_signal.load_data", side_effect=load), \
                 patch("scripts.diagnose_i3_fused_target_signal.load_frozen_b1",
                       return_value=(None, torch.ones(2, 32))), \
                 patch("scripts.diagnose_i3_fused_target_signal.ResNet", TinyNet):
                report = analyze(options)
            self.assertEqual(report["training"]["count"], 2)
            self.assertTrue((root / "report" / "fused_target_signal.json").is_file())
            csv_rows = (root / "report" / "per_image_training.csv").read_text().splitlines()
            self.assertEqual(len(csv_rows), 3)


if __name__ == "__main__":
    unittest.main()
