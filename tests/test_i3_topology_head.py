"""Topology invariance, stable coding loss and training-only pilot protocol."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.diagnose_i3_validation import score_validation
from scripts.run_i3_topology_head_pilot import analyze, validate_cache
from utils.semantic_mixup import frozen_batchnorm
from utils.topology_head import (SmallHashHead, class_targets, coding_rate_loss,
                                 fit_head, topology_loss, topology_weights)


class TopologyHeadTests(unittest.TestCase):
    def test_rate_prefers_diverse_codes_and_has_finite_gradients(self):
        diverse = torch.eye(4, requires_grad=True)
        collapsed = torch.ones(4, 4)
        loss = coding_rate_loss(diverse)
        self.assertLess(loss, coding_rate_loss(collapsed))
        loss.backward()
        self.assertTrue(torch.isfinite(diverse.grad).all())

    def test_batch_targets_use_class_means_without_teacher_gradients(self):
        logits = torch.tensor([[2., -1.], [1., -2.], [-2., 1.]], requires_grad=True)
        targets = class_targets(logits, torch.tensor([0, 0, 1]), 2)
        self.assertEqual(targets.tolist(), [[1., 0.], [1., 0.], [0., 1.]])
        self.assertFalse(targets.requires_grad)

    def test_topology_allows_common_bit_permutation_and_sign_changes(self):
        centers = torch.tensor([[1., 1., -1.], [-1., 1., 1.], [1., -1., 1.]])
        logits = torch.tensor([[.8, -.4, .2], [-.5, .2, .7], [.3, .6, -.2]])
        neighbors = torch.tensor([[1], [2], [0]])
        weights = topology_weights(neighbors, 3, True)
        transformed = logits[:, [2, 0, 1]] * torch.tensor([-1., 1., -1.])
        self.assertAlmostEqual(float(topology_loss(logits, centers, weights)),
                               float(topology_loss(transformed, centers, weights)), places=7)
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertTrue(torch.equal(weights.diagonal(), torch.zeros(3)))

    def test_prototype_pass_preserves_batchnorm_running_statistics(self):
        head = SmallHashHead(4, 3).train()
        head(torch.randn(8, 4))
        before = {key: value.clone() for key, value in head.balance.state_dict().items()}
        with frozen_batchnorm(head):
            head(torch.randn(3, 4)).square().mean().backward()
        self.assertTrue(head.balance.training)
        self.assertTrue(torch.equal(before["running_mean"], head.balance.running_mean))
        self.assertTrue(torch.equal(before["running_var"], head.balance.running_var))

    def test_paired_fit_is_reproducible(self):
        generator = torch.Generator().manual_seed(2)
        features = torch.randn(8, 2, 4, generator=generator)
        labels = torch.arange(8) % 2
        centers = torch.tensor([[1., -1.], [-1., 1.]])
        neighbors = torch.tensor([[1], [0]])
        a, history, fit = fit_head(features, labels, centers, neighbors, "cv_control", 60, "cpu", batch_size=4)
        b, _, _ = fit_head(features, labels, centers, neighbors, "cv_control", 60, "cpu", batch_size=4)
        self.assertEqual(len(history), 5)
        self.assertEqual(fit["selected_epoch"], 5)
        self.assertTrue(all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items()))

    def test_cache_rejects_changed_identity(self):
        with self.assertRaises(ValueError):
            validate_cache({"identity": {"model": "wrong"}}, {"model": "right"}, {}, 32)

    def test_complete_pilot_never_uses_held_out_features_for_training(self):
        class NoQuery:
            def __iter__(self):
                raise AssertionError("Final query was iterated")

        class TinyNet(torch.nn.Module):
            def __init__(self, bits, pretrained=False):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

        class TinyDataset:
            def __init__(self, indices, labels):
                self.indices, self.labels = indices, labels

            def __len__(self):
                return len(self.indices)

        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
        generator = torch.Generator().manual_seed(3)
        training = dict(features=torch.randn(4, 2, 512, generator=generator),
                        classes=torch.tensor(labels[:4]), ids=torch.arange(4))
        centers = torch.tensor([[1., -1.], [-1., 1.]]).repeat(1, 16)
        v_codes = centers.to(torch.int8).numpy()
        d_codes = np.concatenate((v_codes, v_codes))
        expected = {metric: float(values.mean()) for metric, values in
                    score_validation(v_codes, d_codes, labels[4:6], labels[6:])["ap"].items()}
        val = dict(features=torch.randn(2, 512, generator=generator).numpy(),
                   codes=v_codes, classes=labels[4:6], ids=np.arange(4, 6))
        db = dict(features=torch.randn(4, 512, generator=generator).numpy(),
                  codes=d_codes, classes=labels[6:], ids=np.arange(6, 10))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            for name in ("train", "test", "meta"):
                (raw / name).touch()
            checkpoint = root / "model.pt"
            torch.save(TinyNet(32).state_dict(), checkpoint)
            config = dict(variant="b2nfd", code_length=32, seed=60, nfd_alpha=.75,
                          nfd_weight=.01,
                          split_hash="synthetic", root=str(raw), num_workers=0,
                          num_classes=2, batch_size=2)
            run = dict(config=config, path=root, checkpoint=checkpoint,
                       checkpoint_sha256="model", summary={"selected_epoch": 300},
                       selected={"validation": expected})
            datasets = [TinyDataset(indices=np.arange(a, b), labels=labels)
                        for a, b in ((0, 4), (4, 6), (6, 10))]
            loaders = [SimpleNamespace(dataset=dataset) for dataset in datasets]

            def load(args):
                args.split_hash = "synthetic"
                args.dataset_content_sha256 = "data"
                return loaders[0], loaders[0], loaders[1], NoQuery(), loaders[2]

            def fit(features, classes, *args, **kwargs):
                self.assertTrue(torch.equal(features, training["features"]))
                self.assertTrue(torch.equal(classes, training["classes"]))
                return fit_head(features, classes, *args, epochs=1, batch_size=4, **kwargs)

            options = argparse.Namespace(nfd_run=str(root), output_dir=str(root / "pilot"),
                                         data_root=None, cpu=True, gpu=0, num_workers=0,
                                         dry_run=False, feature_cache=None)
            with patch("scripts.run_i3_topology_head_pilot.checked_run", return_value=run), \
                 patch("scripts.run_i3_topology_head_pilot.load_data", side_effect=load), \
                 patch("scripts.run_i3_topology_head_pilot.load_frozen_b1", return_value=(None, centers)), \
                 patch("scripts.run_i3_topology_head_pilot.load_nfd_geometry",
                       return_value=(None, torch.tensor([[1], [0]]), "prototypes")), \
                 patch("scripts.run_i3_topology_head_pilot.ResNet", side_effect=TinyNet) as constructor, \
                 patch("scripts.run_i3_topology_head_pilot.encode_training", return_value=training) as train_encoder, \
                 patch("scripts.run_i3_topology_head_pilot.encode", side_effect=[val, db]) as held_out_encoder, \
                 patch("scripts.run_i3_topology_head_pilot.fit_head", side_effect=fit) as fitter:
                report = analyze(options)
                options.feature_cache = str(root / "pilot" / "frozen_features.pt")
                options.output_dir = str(root / "reuse")
                reused = analyze(options)
            self.assertEqual(fitter.call_count, 8)
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(train_encoder.call_count, 1)
            self.assertEqual(held_out_encoder.call_count, 2)
            self.assertEqual(report["results"], reused["results"])
            self.assertFalse(report["query_evaluated"])
            self.assertEqual(len(report["results"]), 5)
            self.assertTrue((root / "pilot" / "frozen_features.pt").is_file())
            self.assertTrue((root / "pilot" / "topology_head_pilot.json").is_file())
            self.assertTrue((root / "pilot" / "per_query_validation.csv").is_file())


if __name__ == "__main__":
    unittest.main()
