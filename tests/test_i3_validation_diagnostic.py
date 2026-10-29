"""Validation-only diagnostic checks on tiny synthetic codes and runs."""
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

from scripts.diagnose_i3_validation import analyze, checked_run, score_validation
from utils.experiment import selection_receipt
from utils.tools import CalcTopMapPerQuery


class ValidationDiagnosticTests(unittest.TestCase):
    def test_stable_hamming_ap_matches_existing_evaluator_with_ties(self):
        query = np.array([[1, 1, 1, 1], [-1, 1, -1, 1]], dtype=np.int8)
        database = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1],
                             [-1, 1, -1, 1], [1, -1, 1, -1]], dtype=np.int8)
        query_class = np.array([0, 1])
        database_class = np.array([1, 0, 1, 0, 0])
        result = score_validation(query, database, query_class, database_class)
        q_labels = np.eye(2, dtype=np.float32)[query_class]
        d_labels = np.eye(2, dtype=np.float32)[database_class]
        original = CalcTopMapPerQuery(query, q_labels, database, d_labels, [-1, 100, 1000])
        for name, cutoff in (("mAP@ALL", -1), ("mAP@100", 100), ("mAP@1000", 1000)):
            np.testing.assert_allclose(result["ap"][name], original[str(cutoff)])
        self.assertEqual(result["boundary_ties"].tolist(), [3, 1])
        self.assertFalse(result["boundary_extends"].any())
        self.assertEqual(result["false_top100"].tolist(), [2, 3])

    def test_best_epoch_and_receipt_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            config = dict(variant="b3fcd", dataset="cifar-100-new-seg", num_classes=100,
                          topK=[-1, 100, 1000], paired_stage_seeds=True, epoch=3)
            history = [dict(epoch=i, validation=dict(
                **{"mAP@ALL": score, "mAP@100": .6, "mAP@1000": .55}))
                for i, score in enumerate((.4, .5, .45), 1)]
            (run / "config.json").write_text(json.dumps(config))
            (run / "training_history.json").write_text(json.dumps(history))
            (run / "best_model_state.pt").write_bytes(b"synthetic checkpoint")
            summary = dict(selected_epoch=2, selection_metric="validation mAP@ALL",
                           **{"best_validation_mAP@ALL": .5}, **selection_receipt(run))
            (run / "metrics.json").write_text(json.dumps(summary))
            self.assertEqual(checked_run(run)["selected"]["epoch"], 2)
            summary["selected_epoch"] = 3
            (run / "metrics.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "Selected model"):
                checked_run(run)

    def test_complete_diagnostic_reads_validation_and_database_only(self):
        q_codes = np.array([[1, 1, 1, 1], [-1, 1, -1, 1]], dtype=np.int8)
        d_codes = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1],
                            [-1, 1, -1, 1], [1, -1, 1, -1]], dtype=np.int8)
        q_ids, d_ids = np.array([0, 1]), np.arange(2, 7)
        labels = np.array([0, 1, 1, 0, 1, 0, 0])

        class TinyDataset(Dataset):
            def __init__(self, ids, codes):
                self.indices, self.codes, self.labels = ids, codes, labels

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                index = int(self.indices[item])
                target = torch.zeros(100)
                target[self.labels[index]] = 1
                return torch.tensor(self.codes[item], dtype=torch.float32), target, torch.tensor(index)

        class NoQuery:
            def __iter__(self):
                raise AssertionError("Final query loader was iterated")

        class TinyNet(nn.Module):
            def __init__(self, bits, pretrained=False):
                super().__init__()
                self.bits = bits

            def forward(self, images):
                return images

        val = DataLoader(TinyDataset(q_ids, q_codes), batch_size=2)
        db = DataLoader(TinyDataset(d_ids, d_codes), batch_size=2)
        expected = {key: float(value.mean()) for key, value in
                    score_validation(q_codes, d_codes, labels[q_ids], labels[d_ids])["ap"].items()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            config = dict(variant="b2nfd", dataset="cifar-100-new-seg", num_classes=100,
                          topK=[-1, 100, 1000], paired_stage_seeds=True, epoch=1,
                          code_length=4, seed=60, val_per_class=10, resize_size=256,
                          crop_size=224, split_hash="synthetic", root=str(raw))
            paths = []
            for variant in ("b2nfd", "b3fcd"):
                run = root / variant
                run.mkdir()
                config["variant"] = variant
                (run / "config.json").write_text(json.dumps(config))
                (run / "training_history.json").write_text(json.dumps(
                    [dict(epoch=1, validation=expected)]))
                torch.save({}, run / "best_model_state.pt")
                summary = dict(selected_epoch=1, selection_metric="validation mAP@ALL",
                               **{"best_validation_mAP@ALL": expected["mAP@ALL"]},
                               **selection_receipt(run))
                (run / "metrics.json").write_text(json.dumps(summary))
                paths.append(run)

            def fake_load(args):
                args.split_hash = "synthetic"
                args.dataset_content_sha256 = "synthetic-pixels"
                return None, None, val, NoQuery(), db, 0, 0, len(val.dataset), len(db.dataset)

            options = argparse.Namespace(baseline_run=str(paths[0]), candidate_run=str(paths[1]),
                                         output_dir=root / "diagnostic", data_root=None,
                                         cpu=True, gpu=0, num_workers=0, dry_run=False)
            with patch("scripts.diagnose_i3_validation.load_data", side_effect=fake_load), \
                 patch("scripts.diagnose_i3_validation.ResNet", TinyNet):
                report = analyze(options)
            self.assertEqual(report["protocol"],
                             "validation vs original database; final query never iterated")
            self.assertAlmostEqual(report["paired_bootstrap_95"]["mAP@ALL"]["delta"], 0.)
            self.assertTrue((options.output_dir / "per_query_validation.csv").is_file())
            self.assertTrue((options.output_dir / "per_class_validation.csv").is_file())
            self.assertTrue((options.output_dir / "diagnostic.json").is_file())


if __name__ == "__main__":
    unittest.main()
