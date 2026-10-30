"""Readout extrapolation, matched controls and training-only fitting protocol."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.diagnose_i3_validation import score_validation
from scripts.run_i3_readout_screen import analyze
from utils.semantic_readout import fit_readout, readout_logits


class SemanticReadoutTests(unittest.TestCase):
    def geometry(self):
        return (torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]]),
                torch.tensor([0, 1, 0, 1]),
                torch.tensor([[1., -1.], [-1., 1.]]),
                torch.tensor([[0., 1.], [0., -1.]]),
                torch.tensor([[1], [0]], dtype=torch.long))

    def test_virtual_geometry_supplies_mapping_in_unobserved_feature_direction(self):
        x, y, centers, prototypes, near = self.geometry()
        ridge = fit_readout(x, y, centers, prototypes, near, "none")
        graph = fit_readout(x, y, centers, prototypes, near, "nfd")
        unseen = torch.tensor([[0., 1.]])
        self.assertTrue(torch.allclose(readout_logits(unseen, ridge), torch.zeros(1, 2).double()))
        self.assertGreater(readout_logits(unseen, graph)[0, 0], 0)
        self.assertLess(readout_logits(unseen, graph)[0, 1], 0)

    def test_global_and_nfd_controls_agree_when_graphs_are_identical(self):
        x, y, centers, prototypes, near = self.geometry()
        global_head = fit_readout(x, y, centers, prototypes, near, "global")
        nfd_head = fit_readout(x, y, centers, prototypes, near, "nfd")
        self.assertTrue(torch.allclose(global_head["weight"], nfd_head["weight"]))
        self.assertTrue(torch.allclose(global_head["bias"], nfd_head["bias"]))

    def test_positive_feature_rescaling_does_not_change_fitted_policy(self):
        x, y, centers, prototypes, near = self.geometry()
        original = fit_readout(x, y, centers, prototypes, near, "nfd")
        scaled = fit_readout(x * torch.tensor([[2.], [3.], [4.], [5.]]),
                             y, centers, prototypes, near, "nfd")
        self.assertTrue(torch.allclose(original["weight"], scaled["weight"]))
        self.assertTrue(torch.allclose(readout_logits(x, original),
                                       readout_logits(x * 3, original)))

    def test_rejects_nonbinary_targets(self):
        x, y, centers, prototypes, near = self.geometry()
        centers[0, 0] = 0
        with self.assertRaises(ValueError):
            fit_readout(x, y, centers, prototypes, near)

    def test_full_screen_fits_only_training_and_never_iterates_query(self):
        class NoQuery:
            def __iter__(self):
                raise AssertionError("Final query was iterated")

        class TinyNet(torch.nn.Module):
            def __init__(self, bits, pretrained=False):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

        x, y, _, prototypes, near = self.geometry()
        center_rows = torch.tensor([[1., -1.], [-1., 1.]]).repeat(1, 16)
        val_codes = center_rows.to(torch.int8).numpy()
        db_codes = np.concatenate((val_codes, val_codes))
        val_classes, db_classes = np.array([0, 1]), np.array([0, 1, 1, 0])
        expected = {metric: float(values.mean()) for metric, values in
                    score_validation(val_codes, db_codes, val_classes, db_classes)["ap"].items()}
        training = dict(features=x.numpy(), classes=y.numpy(), ids=np.array([10, 11, 12, 13]))
        val = dict(features=prototypes.numpy(), codes=val_codes,
                   classes=val_classes, ids=np.array([20, 21]))
        db = dict(features=np.concatenate((prototypes.numpy(), prototypes.numpy())),
                  codes=db_codes, classes=db_classes, ids=np.array([30, 31, 32, 33]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            checkpoint = root / "model.pt"
            torch.save(TinyNet(32).state_dict(), checkpoint)
            config = dict(variant="b2nfd", code_length=32, seed=60, nfd_alpha=.75,
                          nfd_weight=.01, split_hash="test-split", root=str(raw),
                          num_workers=0, num_classes=2, nfd_basis_sha256="test-basis")
            run = dict(config=config, path=root, checkpoint=checkpoint,
                       checkpoint_sha256="test-model", summary={"selected_epoch": 300},
                       selected={"validation": expected})
            relation_loader = SimpleNamespace(dataset=[0, 1, 2, 3])
            val_loader, db_loader = SimpleNamespace(dataset=[0, 1]), SimpleNamespace(dataset=[0, 1, 2, 3])

            def load(args):
                args.split_hash = "test-split"
                args.dataset_content_sha256 = "test-data"
                return None, relation_loader, val_loader, NoQuery(), db_loader

            def fit(features, labels, *args):
                np.testing.assert_array_equal(features, training["features"])
                np.testing.assert_array_equal(labels, training["classes"])
                return fit_readout(features, labels, *args)

            options = argparse.Namespace(nfd_run=str(root), output_dir=str(root / "pilot"),
                                         data_root=None, cpu=True, gpu=0, num_workers=0,
                                         dry_run=False)
            with patch("scripts.run_i3_readout_screen.checked_run", return_value=run), \
                 patch("scripts.run_i3_readout_screen.load_data", side_effect=load), \
                 patch("scripts.run_i3_readout_screen.load_frozen_b1", return_value=(None, center_rows)), \
                 patch("scripts.run_i3_readout_screen.load_nfd_geometry",
                       return_value=(prototypes, near, "test-prototypes")), \
                 patch("scripts.run_i3_readout_screen.ResNet", TinyNet), \
                 patch("scripts.run_i3_readout_screen.encode", side_effect=[training, val, db]) as encoder, \
                 patch("scripts.run_i3_readout_screen.fit_readout", side_effect=fit) as fitter:
                report = analyze(options)
            self.assertEqual(fitter.call_count, 3)
            self.assertEqual([call.args[0] for call in encoder.call_args_list],
                             [relation_loader, val_loader, db_loader])
            self.assertEqual(report["results"]["original"]["mAP@ALL"], expected["mAP@ALL"])
            self.assertFalse(report["query_evaluated"])
            self.assertTrue((root / "pilot" / "readout_heads.pt").is_file())
            self.assertTrue((root / "pilot" / "readout_screen.json").is_file())
            self.assertEqual(len((root / "pilot" / "summary.csv").read_text().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
