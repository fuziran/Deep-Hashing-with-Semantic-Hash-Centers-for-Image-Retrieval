"""CPU unit checks only: synthetic tensors, no dataset or pretrained weights."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from network import ResNet
from tests.test_i1_relation import args_for
from utils.coarse_aux import (auxiliary_weight, hierarchy_errors, load_fixed_centers,
                              make_auxiliary_head, mapping_from_labels)
from utils.experiment import build_cache_metadata, save_cache


def tiny_net():
    net = ResNet.__new__(ResNet)
    nn.Module.__init__(net)
    net.feature_layers = nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1))
    net.hash_layer = nn.Linear(4, 8)
    return net


class CoarseAuxTests(unittest.TestCase):
    def test_mapping_is_by_fine_id_not_sample_order(self):
        fine = np.tile(np.arange(100), 2)[::-1]
        coarse = fine // 5
        self.assertEqual(mapping_from_labels(fine, coarse), (np.arange(100) // 5).tolist())
        coarse[0] = 0
        with self.assertRaises(ValueError):
            mapping_from_labels(fine, coarse)
        with self.assertRaises(ValueError):
            mapping_from_labels(np.arange(99), np.arange(99) // 5)
        with self.assertRaises(ValueError):
            mapping_from_labels(np.arange(100), np.zeros(100, dtype=int))

    def test_schedule(self):
        self.assertAlmostEqual(auxiliary_weight(.05, 10, 0), .005)
        self.assertAlmostEqual(auxiliary_weight(.05, 10, 9), .05)
        self.assertAlmostEqual(auxiliary_weight(.05, 10, 20), .05)
        self.assertEqual(auxiliary_weight(0, 10, 0), 0)
        self.assertEqual(auxiliary_weight(.05, 0, 0), .05)
        with self.assertRaises(ValueError):
            auxiliary_weight(float("nan"), 10, 0)

    def test_shared_gradients_inference_and_rng(self):
        net = tiny_net()
        images = torch.randn(2, 3, 2, 2)
        rng = torch.get_rng_state().clone()
        self.assertIsNone(make_auxiliary_head(net, 0, "coarse"))
        head = make_auxiliary_head(net, .05, "coarse")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        original = net(images)
        codes, features = net.forward_with_features(images)
        torch.testing.assert_close(original, codes, rtol=0, atol=0)
        nn.functional.cross_entropy(head(features), torch.tensor([0, 19])).backward()
        self.assertGreater(net.feature_layers[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(net.hash_layer.weight.grad)
        self.assertFalse(any("aux" in key for key in net.state_dict()))
        self.assertEqual(make_auxiliary_head(net, .05, "fine").out_features, 100)

    def test_zero_aux_matches_legacy_update(self):
        from train import CSQLoss
        net = tiny_net()
        legacy = copy.deepcopy(net)
        images = torch.randn(3, 3, 2, 2)
        labels = torch.eye(100)[[0, 3, 9]]
        args = SimpleNamespace(device=torch.device("cpu"), lambd=1e-4)
        centers = torch.randint(0, 2, (100, 8)).float() * 2 - 1
        criterion = CSQLoss(args, 8, centers)
        new_optimizer = torch.optim.RMSprop(net.parameters(), lr=7e-5, weight_decay=1e-5)
        old_optimizer = torch.optim.RMSprop(legacy.parameters(), lr=7e-5, weight_decay=1e-5)
        self.assertIsNone(make_auxiliary_head(net, 0, "coarse"))
        loss = criterion(net(images), labels, None, args)
        features = legacy.feature_layers(images)
        old_output = legacy.hash_layer(features.view(features.size(0), -1))
        old_loss = criterion(old_output, labels, None, args)
        torch.testing.assert_close(loss, old_loss, rtol=0, atol=0)
        loss.backward()
        old_loss.backward()
        new_optimizer.step()
        old_optimizer.step()
        for key, value in net.state_dict().items():
            torch.testing.assert_close(value, legacy.state_dict()[key], rtol=0, atol=0)

    def test_cli_rejects_unfrozen_or_invalid_auxiliary_runs(self):
        from run import load_config
        import contextlib
        import io
        for options in (["--aux-weight", "0.05"], ["--aux-weight", "nan"],
                        ["--aux-warmup-epochs", "-1"], ["--epoch", "0"],
                        ["--variant", "b1", "--center-cache", "H1.pt"]):
            with patch("sys.argv", ["run.py", "--cpu"] + options), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                load_config()
        with patch("sys.argv", ["run.py", "--cpu", "--variant", "b1", "--stage", "train",
                                "--center-cache", "H1.pt", "--similarity-cache", "S1.pt",
                                "--aux-weight", "0.05"]):
            self.assertEqual(load_config().aux_weight, .05)

    def test_fixed_cache_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory, similarity_hash="S1", similarity_content_sha256="digest")
            metadata = build_cache_metadata(args, "centers")
            centers = torch.ones(4, 3)
            path = str(Path(directory) / "centers.pt")
            save_cache(path, centers, metadata)
            args.git_commit = "new-stage3-code"
            torch.testing.assert_close(load_fixed_centers(args, path), centers)
            args.seed += 1
            with self.assertRaises(RuntimeError):
                load_fixed_centers(args, path)
            args.seed -= 1
            args.similarity_content_sha256 = "other-S1"
            with self.assertRaises(RuntimeError):
                load_fixed_centers(args, path)
            args.similarity_content_sha256 = "digest"
            save_cache(path, centers * 0, metadata)
            with self.assertRaises(RuntimeError):
                load_fixed_centers(args, path)
            bad = dict(metadata, seed=99)
            save_cache(path, centers, bad)
            with self.assertRaises(RuntimeError):
                load_fixed_centers(args, path)

    def test_hierarchy_diagnostic_uses_stable_ties(self):
        result = hierarchy_errors(np.ones((1, 2)), np.eye(100)[[0]],
                                  np.ones((3, 2)), np.eye(100)[[1, 5, 0]],
                                  (np.arange(100) // 5).tolist(), topks=(1, 3))
        self.assertEqual(result["top1"]["same_coarse_wrong_fine_count"], 1)
        self.assertEqual(result["top3"]["cross_coarse_error_count"], 1)
        self.assertEqual(result["top3"]["retrieved_count"], 3)

    def test_validation_only_and_inference_checkpoint(self):
        from train import train_val
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(code_length=8, num_classes=100, device=torch.device("cpu"),
                                   net=lambda bit: tiny_net(), lr=1e-4, epoch=1, test_map=1,
                                   lambd=1e-4, aux_weight=.05, aux_target="coarse",
                                   aux_warmup_epochs=10, fine_to_coarse=(np.arange(100)//5).tolist(),
                                   validation_only=True, run_dir=directory, info="unit-test")
            batch = [(torch.randn(2, 3, 2, 2), torch.eye(100)[[0, 5]], torch.arange(2))]
            val, query, database = object(), object(), object()
            metrics = {"mAP@ALL": .2, "mAP@100": .3, "mAP@1000": .4}
            with patch("train._evaluate", return_value=(metrics, None, None, None, None, None)) as evaluate, \
                    patch("train._save_curves"):
                result = train_val(args, torch.ones(100, 8), batch, val, query, database, 3)
            self.assertEqual(evaluate.call_count, 1)
            self.assertIs(evaluate.call_args.args[0], val)
            self.assertFalse(result["query_evaluated"])
            self.assertNotIn("query_metrics", result)
            self.assertFalse((Path(directory)/"retrieval_outputs.pt").exists())
            state = torch.load(Path(directory)/"best_model_state.pt", weights_only=True)
            tiny_net().load_state_dict(state, strict=True)
            self.assertTrue((Path(directory)/"best_auxiliary_head.pt").exists())


if __name__ == "__main__":
    unittest.main()
