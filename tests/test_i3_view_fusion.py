"""Frozen two-view screen preserves original evaluation and query isolation."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.diagnose_i3_validation import score_validation
from scripts.diagnose_i3_view_fusion import analyze, binarize


class ViewFusionTests(unittest.TestCase):
    def test_zero_logit_uses_original_positive_sign_convention(self):
        self.assertEqual(binarize(np.array([[0., -1., 1.]]))[0].tolist(),
                         [1, -1, 1])

    def test_frozen_screen_never_iterates_final_query(self):
        codes = np.tile(np.array([[1, 1, -1, -1], [-1, -1, 1, 1],
                                  [1, 1, -1, -1], [-1, -1, 1, 1],
                                  [1, 1, -1, -1], [-1, -1, 1, 1]],
                                 dtype=np.float32), (1, 8))
        classes = np.array([0, 1, 0, 1, 1, 0])

        class TinyDataset(Dataset):
            def __init__(self, ids):
                self.indices, self.labels = ids, classes

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                image_id = int(self.indices[item])
                return (torch.tensor(codes[image_id]).reshape(1, 1, 32),
                        torch.nn.functional.one_hot(
                            torch.tensor(classes[image_id]), 100).float(),
                        torch.tensor(image_id))

        class NoQuery:
            def __iter__(self):
                raise AssertionError("Final query was iterated")

        class TinyNet(nn.Module):
            def __init__(self, bits, pretrained=False):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(()))

            def forward(self, image):
                return image.flatten(1) * self.scale

        val = DataLoader(TinyDataset(np.array([0, 1])), batch_size=2)
        db = DataLoader(TinyDataset(np.array([2, 3, 4, 5])), batch_size=2)
        expected = {name: float(values.mean()) for name, values in
                    score_validation(codes[:2].astype(np.int8), codes[2:].astype(np.int8),
                                     classes[:2], classes[2:])["ap"].items()}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            run = root / "run"
            run.mkdir()
            config = dict(variant="b2nfd", dataset="cifar-100-new-seg", num_classes=100,
                          topK=[-1, 100, 1000], paired_stage_seeds=True, epoch=1,
                          code_length=32, seed=60, val_per_class=10, resize_size=256,
                          crop_size=224, split_hash="synthetic", root=str(raw))
            (run / "config.json").write_text(json.dumps(config))
            (run / "training_history.json").write_text(json.dumps(
                [dict(epoch=1, validation=expected)]))
            torch.save(TinyNet(32).state_dict(), run / "best_model_state.pt")
            (run / "metrics.json").write_text(json.dumps(
                dict(selected_epoch=1, selection_metric="validation mAP@ALL",
                     **{"best_validation_mAP@ALL": expected["mAP@ALL"]})))

            def fake_load(args):
                args.split_hash = "synthetic"
                args.dataset_content_sha256 = "synthetic-pixels"
                return None, None, val, NoQuery(), db, 0, 0, len(val.dataset), len(db.dataset)

            options = argparse.Namespace(nfd_run=str(run), output_dir=str(root / "screen"),
                                         data_root=None, gpu=0, cpu=True,
                                         num_workers=0, dry_run=False)
            with patch("scripts.diagnose_i3_view_fusion.load_data", side_effect=fake_load), \
                 patch("scripts.diagnose_i3_view_fusion.ResNet", TinyNet):
                report = analyze(options)
            self.assertAlmostEqual(report["comparison"]["symmetric_fusion"]
                                   ["mAP@ALL"]["baseline"], expected["mAP@ALL"])
            self.assertTrue((root / "screen" / "view_fusion.json").is_file())
            self.assertTrue((root / "screen" / "per_query_validation.csv").is_file())


if __name__ == "__main__":
    unittest.main()
