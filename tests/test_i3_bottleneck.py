"""Small protocol checks for the frozen representation audit."""
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

from scripts.diagnose_i3_bottleneck import analyze, score_cosine
from scripts.diagnose_i3_validation import score_validation
from utils.experiment import selection_receipt


class BottleneckTests(unittest.TestCase):
    def test_cosine_ap_uses_stable_ties_and_original_cutoffs(self):
        q = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        d = np.array([[1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
        classes = np.array([1, 0, 0])
        scored = score_cosine(q, d, np.array([0, 1]), classes, torch.device("cpu"))
        self.assertAlmostEqual(scored["ap"]["mAP@ALL"][0], 7 / 12)
        self.assertAlmostEqual(scored["ap"]["mAP@ALL"][1], .5)
        self.assertEqual(scored["false_top100"].tolist(), [1, 2])

    def test_audit_never_iterates_final_query(self):
        q_codes = np.array([[1, 1, -1, -1], [-1, -1, 1, 1]], dtype=np.int8)
        d_codes = np.array([[1, 1, -1, -1], [-1, -1, 1, 1],
                            [1, 1, -1, -1], [-1, -1, 1, 1]], dtype=np.int8)
        all_codes = np.concatenate((q_codes, d_codes))
        classes = np.array([0, 1, 0, 1, 1, 0])

        class TinyDataset(Dataset):
            def __init__(self, ids):
                self.indices, self.labels = ids, classes

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                index = int(self.indices[item])
                return (torch.tensor(all_codes[index], dtype=torch.float32),
                        torch.nn.functional.one_hot(torch.tensor(classes[index]), 100).float(),
                        torch.tensor(index))

        class NoQuery:
            def __iter__(self):
                raise AssertionError("Final query was iterated")

        class TinyNet(nn.Module):
            def __init__(self, bits, pretrained=False):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(()))

            def forward_with_features(self, images):
                return images * self.scale, images * self.scale

        val = DataLoader(TinyDataset(np.array([0, 1])), batch_size=2)
        db = DataLoader(TinyDataset(np.array([2, 3, 4, 5])), batch_size=2)
        expected = {name: float(values.mean()) for name, values in
                    score_validation(q_codes, d_codes, classes[:2], classes[2:])["ap"].items()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            run = root / "run"
            run.mkdir()
            config = dict(variant="b2nfd", dataset="cifar-100-new-seg", num_classes=100,
                          topK=[-1, 100, 1000], paired_stage_seeds=True, epoch=1,
                          code_length=4, seed=60, val_per_class=10, resize_size=256,
                          crop_size=224, split_hash="synthetic", root=str(raw))
            (run / "config.json").write_text(json.dumps(config))
            (run / "training_history.json").write_text(json.dumps(
                [dict(epoch=1, validation=expected)]))
            torch.save(TinyNet(4).state_dict(), run / "best_model_state.pt")
            summary = dict(selected_epoch=1, selection_metric="validation mAP@ALL",
                           **{"best_validation_mAP@ALL": expected["mAP@ALL"]},
                           **selection_receipt(run))
            (run / "metrics.json").write_text(json.dumps(summary))

            def fake_load(args):
                args.split_hash = "synthetic"
                args.dataset_content_sha256 = "synthetic-pixels"
                return None, val, val, NoQuery(), db, 0, len(val.dataset), len(val.dataset), len(db.dataset)

            options = argparse.Namespace(run=str(run), output_dir=str(root / "audit"),
                                         data_root=None, gpu=0, cpu=True,
                                         num_workers=0, chunk_size=2, dry_run=False)
            with patch("scripts.diagnose_i3_bottleneck.load_data", side_effect=fake_load), \
                 patch("scripts.diagnose_i3_bottleneck.ResNet", TinyNet):
                report = analyze(options)
            self.assertEqual(report["retrieval"]["binary_hamming"]["mAP@ALL"],
                             expected["mAP@ALL"])
            self.assertEqual(report["horizontal_flip"]["validation_mean_changed_bit_fraction"], 1.)
            self.assertEqual(report["horizontal_flip"]["training_mean_changed_bit_fraction"], 1.)
            self.assertTrue((root / "audit" / "diagnostic.json").is_file())
            self.assertTrue((root / "audit" / "per_query_validation.csv").is_file())


if __name__ == "__main__":
    unittest.main()
