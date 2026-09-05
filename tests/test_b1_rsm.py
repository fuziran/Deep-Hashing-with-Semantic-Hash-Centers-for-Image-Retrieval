import argparse
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from GenerateSimilarityMatrix import (
    build_rsm_similarity,
    load_explicit_classifier_cache,
    normalize_and_symmetrize_similarity,
    prediction_reliability,
    select_temperature,
    weighted_class_average,
)
from data.data_loader import DeterministicWeakMultiViewTransform
from train import validate_frozen_query_protocol
from utils.experiment import build_cache_metadata, save_cache, state_dict_sha256


class TestRsmMath(unittest.TestCase):
    def test_temperature_selection_cannot_be_worse_than_one(self):
        logits = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
        targets = torch.tensor([0, 1, 1])
        temperature, losses = select_temperature(
            logits, targets, [0.7, 1.0, 1.5, 2.0]
        )
        self.assertGreater(temperature, 0)
        self.assertLessEqual(losses[temperature], losses[1.0])

    def test_temperature_grid_must_include_identity(self):
        with self.assertRaises(ValueError):
            select_temperature(torch.ones(2, 2), torch.tensor([0, 1]), [0.7, 1.5])

    def test_reliability_is_bounded_and_tracks_entropy(self):
        probabilities = torch.tensor([[0.5, 0.5], [0.99, 0.01]], dtype=torch.float64)
        reliability = prediction_reliability(probabilities)
        self.assertTrue(torch.all((0 <= reliability) & (reliability <= 1)))
        self.assertLess(reliability[0], reliability[1])

    def test_weighted_class_average_matches_hand_calculation(self):
        values = torch.tensor([[1.0, 0.0], [3.0, 2.0], [4.0, 6.0]])
        targets = torch.tensor([0, 0, 1])
        weights = torch.tensor([1.0, 3.0, 2.0])
        average, totals = weighted_class_average(values, targets, weights, 2)
        expected = torch.tensor([[2.5, 1.5], [4.0, 6.0]])
        self.assertTrue(torch.allclose(average, expected))
        self.assertTrue(torch.equal(totals, torch.tensor([4.0, 2.0])))

    def test_rsm_similarity_invariants(self):
        confusion = torch.tensor(
            [[0.0, 0.8, 0.2], [0.7, 0.0, 0.3], [0.1, 0.4, 0.0]],
            dtype=torch.float64,
        )
        prototypes = torch.tensor(
            [[1.0, 0.2, -0.1], [0.2, 1.0, 0.4], [-0.1, 0.4, 1.0]],
            dtype=torch.float64,
        )
        similarity = build_rsm_similarity(confusion, prototypes, alpha=0.7)
        self.assertTrue(torch.isfinite(similarity).all())
        self.assertTrue(torch.allclose(similarity, similarity.T))
        self.assertTrue(torch.allclose(similarity.diag(), torch.ones(3, dtype=torch.float64)))
        self.assertGreaterEqual(similarity.min().item(), -1.0)
        self.assertLessEqual(similarity.max().item(), 1.0)

    def test_confusion_only_keeps_b0_normalization_order(self):
        confusion = torch.tensor(
            [[0.0, 0.2, 0.8], [0.4, 0.0, 0.6], [0.7, 0.3, 0.0]],
            dtype=torch.float64,
        )
        expected = normalize_and_symmetrize_similarity(confusion)
        actual = build_rsm_similarity(confusion, prototype_similarity=None)
        self.assertTrue(torch.equal(actual, expected))


class TestRsmDeterminism(unittest.TestCase):
    def test_multiview_transform_is_index_deterministic(self):
        pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        image = Image.fromarray(pixels)
        transform = DeterministicWeakMultiViewTransform(
            resize_size=40,
            crop_size=32,
            num_views=4,
            seed=60,
            normalize=transforms.Normalize([0.5] * 3, [0.5] * 3),
        )
        first = transform(image, sample_index=17)
        second = transform(image, sample_index=17)
        other = transform(image, sample_index=18)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))


class TestClassifierReuse(unittest.TestCase):
    def _args(self, seed=60):
        return argparse.Namespace(
            dataset="cifar-100-new-seg",
            seed=seed,
            split_hash="split",
            num_classes=100,
            git_commit="current",
            lr=7e-5,
            classify_epoch=300,
            resize_size=256,
            crop_size=224,
            method="B1",
        )

    def test_state_dict_hash_is_order_independent(self):
        first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
        second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        self.assertEqual(state_dict_sha256(first), state_dict_sha256(second))

    def test_explicit_classifier_allows_only_commit_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "classifier.pt")
            source_args = self._args()
            source_args.git_commit = "b0-commit"
            metadata = build_cache_metadata(source_args, "classifier")
            state = {"weight": torch.tensor([[1.0, 2.0]])}
            save_cache(path, state, metadata)

            current_args = self._args()
            loaded = load_explicit_classifier_cache(current_args, path)
            self.assertTrue(torch.equal(loaded["weight"], state["weight"]))
            self.assertEqual(current_args.classifier_sha256, state_dict_sha256(state))

    def test_explicit_classifier_rejects_split_protocol_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "classifier.pt")
            source_args = self._args()
            metadata = build_cache_metadata(source_args, "classifier")
            save_cache(path, {"weight": torch.ones(1)}, metadata)
            changed = self._args(seed=40)
            with self.assertRaises(RuntimeError):
                load_explicit_classifier_cache(changed, path)

    def test_rsm_hyperparameter_change_invalidates_similarity_cache(self):
        first = self._args()
        first.mask_strategy = "predicted_argmax"
        first.classifier_sha256 = "classifier"
        first.rsm_views = 4
        first.rsm_temperature_grid = [0.7, 1.0, 1.5, 2.0]
        first.rsm_confusion_alpha = 0.7
        second = argparse.Namespace(**vars(first))
        second.rsm_confusion_alpha = 0.9
        first_metadata = build_cache_metadata(first, "similarity")
        second_metadata = build_cache_metadata(second, "similarity")
        self.assertNotEqual(
            first_metadata["config_hash"], second_metadata["config_hash"]
        )


class TestQueryIsolation(unittest.TestCase):
    def _args(self):
        return argparse.Namespace(
            method="B1",
            dataset="cifar-100-new-seg",
            seed=60,
            split_hash="split",
            code_length=32,
            topK=[-1, 100, 1000],
        )

    def _summary(self):
        return {
            "method": "B1",
            "dataset": "cifar-100-new-seg",
            "seed": 60,
            "split_sha256": "split",
            "code_length": 32,
            "topK": [-1, 100, 1000],
            "query_evaluation_count": 0,
        }

    def test_matching_frozen_protocol_is_accepted(self):
        validate_frozen_query_protocol(self._summary(), self._args())

    def test_repeated_query_evaluation_is_rejected(self):
        summary = self._summary()
        summary["query_evaluation_count"] = 1
        with self.assertRaises(RuntimeError):
            validate_frozen_query_protocol(summary, self._args())

    def test_split_change_is_rejected(self):
        args = self._args()
        args.split_hash = "different"
        with self.assertRaises(RuntimeError):
            validate_frozen_query_protocol(self._summary(), args)


if __name__ == "__main__":
    unittest.main()
