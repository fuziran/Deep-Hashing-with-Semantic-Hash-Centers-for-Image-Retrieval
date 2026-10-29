"""Focused checks for the frozen prototype margin screen."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.diagnose_i3_prototype_margin import analyze, inspect, margins


class PrototypeMarginTests(unittest.TestCase):
    def test_margin_uses_nearest_other_class(self):
        prototypes = torch.eye(3)
        features = torch.tensor([[1., .8, 0.], [.8, 1., 0.]])
        scores, nearest = margins(features, prototypes, torch.tensor([0, 1]))
        self.assertEqual(nearest.tolist(), [1, 0])
        torch.testing.assert_close(scores, torch.tensor([.2, .2]) / (1.64 ** .5))

    def test_inspect_covers_only_supplied_training_ids(self):
        class Data(Dataset):
            indices = [7, 9]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.eye(2)[index]
                return image, torch.nn.functional.one_hot(torch.tensor(index), 2).float(), self.indices[index]

        class Student(torch.nn.Module):
            def forward_with_features(self, images):
                return images, images

        class Teacher(torch.nn.Module):
            def forward_features(self, images):
                return images

        result = inspect(DataLoader(Data(), batch_size=2), Student(), Teacher(),
                         torch.eye(2), torch.tensor([[1], [0]]),
                         torch.device("cpu"), "training")
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["teacher_prototype_accuracy"], 1.)

    def test_dry_run_does_not_iterate_validation_or_final_query(self):
        class Forbidden:
            def __iter__(self):
                raise AssertionError("Held-out loader was iterated")

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            run = root / "run"
            run.mkdir()
            config_path = run / "config.json"
            config_path.write_text(json.dumps(dict(
                variant="b2nfd", dataset="cifar-100-new-seg", seed=60,
                code_length=32, nfd_alpha=.75, paired_stage_seeds=True,
                root=str(raw), split_hash="split")))
            (run / "best_model_state.pt").write_bytes(b"checkpoint")

            def fake_load(args):
                args.split_hash = "split"
                args.dataset_content_sha256 = "pixels"
                return Forbidden(), Forbidden(), Forbidden(), Forbidden(), Forbidden(), 0, 0, 0, 0

            options = argparse.Namespace(nfd_config=str(config_path),
                                         output_dir=str(root / "out"), data_root=None,
                                         cpu=True, gpu=0, num_workers=0,
                                         dry_run=True)
            original_tempdir = tempfile.TemporaryDirectory
            with patch("scripts.diagnose_i3_prototype_margin.load_data", side_effect=fake_load), \
                 patch("scripts.diagnose_i3_prototype_margin.tempfile.TemporaryDirectory",
                       side_effect=lambda **kwargs: original_tempdir(dir=root)):
                report = analyze(options)
            self.assertTrue(report["dry_run"])
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
