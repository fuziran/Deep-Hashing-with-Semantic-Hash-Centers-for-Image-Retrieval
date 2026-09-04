import argparse
import os
import tempfile
import unittest

import numpy as np
import torch

from GenerateSemanticHashCenters import (
    keep_better_centers,
    loss_is_not_worse,
    pair_quadratic_gradient,
    pairwise_hamming_distances,
    select_monotonic_discrete_update,
    semantic_similarity_loss,
)
from GenerateSimilarityMatrix import (
    load_explicit_similarity_cache,
    normalize_and_symmetrize_similarity,
)
from data.data_loader import _split_sha256, _stratified_split_cifar100
from utils.experiment import build_cache_metadata, load_cache, save_cache
from utils.tools import _calc_map_for_one_topk


class TestCifarSplit(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(100, dtype=np.int64), 600)

    def test_counts_disjointness_and_class_balance(self):
        train, validation, query, database = _stratified_split_cifar100(
            self.labels, seed=60, val_per_class=10
        )
        self.assertEqual((len(train), len(validation), len(query), len(database)),
                         (9000, 1000, 5000, 45000))
        combined = np.concatenate((train, validation, query, database))
        self.assertEqual(len(np.unique(combined)), 60000)
        for indices, expected in (
            (train, 90), (validation, 10), (query, 50), (database, 450)
        ):
            counts = np.bincount(self.labels[indices], minlength=100)
            np.testing.assert_array_equal(counts, np.full(100, expected))

    def test_same_seed_same_hash(self):
        first = _stratified_split_cifar100(self.labels, seed=60, val_per_class=10)
        second = _stratified_split_cifar100(self.labels, seed=60, val_per_class=10)
        self.assertEqual(_split_sha256(*first), _split_sha256(*second))

    def test_paper_protocol_counts(self):
        train, validation, query, database = _stratified_split_cifar100(
            self.labels, seed=60, val_per_class=0
        )
        self.assertEqual(
            (len(train), len(validation), len(query), len(database)),
            (10000, 0, 5000, 45000),
        )
        combined = np.concatenate((train, query, database))
        self.assertEqual(len(np.unique(combined)), 60000)


class TestSimilarityMatrix(unittest.TestCase):
    def test_paper_order_invariants(self):
        class_average = torch.tensor(
            [[0.0, 0.2, 0.8], [0.4, 0.0, 0.6], [0.7, 0.3, 0.0]],
            dtype=torch.float64,
        )
        similarity = normalize_and_symmetrize_similarity(class_average)
        self.assertTrue(torch.allclose(similarity, similarity.T))
        self.assertTrue(
            torch.allclose(
                torch.diag(similarity),
                torch.ones(
                    3, dtype=similarity.dtype, device=similarity.device
                ),
            )
        )
        self.assertTrue(torch.isfinite(similarity).all())


class TestCenterGradient(unittest.TestCase):
    def test_pair_gradient_matches_autograd(self):
        h_i = torch.tensor([0.2, -0.3, 0.7], requires_grad=True)
        h_j = torch.tensor([-0.5, 0.4, 0.1])
        loss = torch.dot(h_i, h_j).pow(2)
        loss.backward()
        expected = h_i.grad
        actual = pair_quadratic_gradient(h_i.detach(), h_j)
        self.assertTrue(torch.allclose(actual, expected))

    def test_raw_center_is_kept_when_candidate_is_worse(self):
        raw = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        candidate = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        similarity = raw.T @ raw / raw.shape[0]
        raw_loss = semantic_similarity_loss(similarity, raw).item()
        candidate_loss = semantic_similarity_loss(similarity, candidate).item()
        best, best_loss, improved = keep_better_centers(
            raw, raw_loss, candidate, candidate_loss
        )
        self.assertFalse(improved)
        self.assertEqual(best_loss, raw_loss)
        self.assertTrue(torch.equal(best, raw))

    def test_pairwise_distance_excludes_diagonal(self):
        centers = torch.tensor(
            [[1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        )
        distances = pairwise_hamming_distances(centers)
        self.assertEqual(distances.numel(), 3)
        self.assertEqual(distances.min().item(), 1.0)

    def test_loss_guard_allows_float32_roundoff_only(self):
        raw_loss = 0.05320427194237709
        cpu_recomputed_loss = 0.05320427939295769
        self.assertTrue(loss_is_not_worse(cpu_recomputed_loss, raw_loss))
        self.assertFalse(loss_is_not_worse(raw_loss + 1e-3, raw_loss))

    def test_monotonic_search_finds_feasible_improvement(self):
        centers = torch.tensor(
            [
                [1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
            ]
        )
        target = centers.clone()
        target[1, 0] = -1
        similarity = target.T @ target / target.shape[0]
        gradient = torch.tensor([0.0, 1.0, 0.0, 0.0])

        candidate, diagnostics = select_monotonic_discrete_update(
            centers, 0, gradient, similarity, min_distance=1
        )
        updated = centers.clone()
        updated[:, 0] = candidate

        self.assertTrue(diagnostics["accepted"])
        updated_loss = semantic_similarity_loss(similarity, updated).item()
        original_loss = semantic_similarity_loss(similarity, centers).item()
        self.assertLess(updated_loss, original_loss)
        self.assertAlmostEqual(
            diagnostics["loss_delta"], updated_loss - original_loss, places=7
        )
        self.assertGreaterEqual(pairwise_hamming_distances(updated).min().item(), 1)

    def test_monotonic_search_rejects_non_improving_candidates(self):
        centers = torch.tensor(
            [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
        )
        similarity = centers.T @ centers / centers.shape[0]
        gradient = torch.tensor([1.0, 0.0, 0.0, 0.0])

        candidate, diagnostics = select_monotonic_discrete_update(
            centers, 0, gradient, similarity, min_distance=1
        )

        self.assertFalse(diagnostics["accepted"])
        self.assertTrue(torch.equal(candidate, centers[:, 0]))
        self.assertGreater(diagnostics["loss_rejections"], 0)

    def test_monotonic_search_preserves_minimum_distance(self):
        centers = torch.tensor(
            [[1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]
        )
        target = centers.clone()
        target[0, 0] = -1
        similarity = target.T @ target / target.shape[0]
        gradient = torch.tensor([1.0, 0.0, 0.0, 0.0])

        candidate, diagnostics = select_monotonic_discrete_update(
            centers, 0, gradient, similarity, min_distance=2
        )

        self.assertFalse(diagnostics["accepted"])
        self.assertTrue(torch.equal(candidate, centers[:, 0]))
        self.assertGreater(diagnostics["constraint_rejections"], 0)


class TestMetrics(unittest.TestCase):
    def test_hand_computed_map(self):
        query_binary = np.ones((1, 4), dtype=np.float32)
        retrieval_binary = np.asarray(
            [
                [1, 1, 1, 1],
                [-1, 1, 1, 1],
                [-1, -1, 1, 1],
                [-1, -1, -1, 1],
                [-1, -1, -1, -1],
            ],
            dtype=np.float32,
        )
        query_label = np.asarray([[1, 0]], dtype=np.float32)
        retrieval_label = np.asarray(
            [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32
        )
        actual = _calc_map_for_one_topk(
            query_binary, query_label, retrieval_binary, retrieval_label, -1
        )
        expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
        self.assertAlmostEqual(actual, expected)


class TestCacheMetadata(unittest.TestCase):
    def _args(self, seed):
        return argparse.Namespace(
            protocol="audited_b0",
            dataset="cifar-100-new-seg",
            seed=seed,
            split_hash="split",
            num_classes=100,
            code_length=32,
            mask_strategy="predicted_argmax",
            git_commit="commit",
            similarity_hash="similarity",
            lr=7e-5,
            classify_epoch=300,
            resize_size=256,
            crop_size=224,
        )

    def test_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.pt")
            metadata = build_cache_metadata(self._args(60), "similarity")
            save_cache(path, torch.ones(2), metadata)
            mismatched = build_cache_metadata(self._args(40), "similarity")
            with self.assertRaises(RuntimeError):
                load_cache(path, mismatched)

    def test_protocol_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.pt")
            audited_args = self._args(60)
            metadata = build_cache_metadata(audited_args, "similarity")
            save_cache(path, torch.ones(2), metadata)
            paper_args = self._args(60)
            paper_args.protocol = "paper_repro"
            mismatched = build_cache_metadata(paper_args, "similarity")
            with self.assertRaises(RuntimeError):
                load_cache(path, mismatched)

    def test_explicit_similarity_allows_only_commit_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "similarity.pt")
            source_args = self._args(60)
            source_args.git_commit = "stage1-commit"
            metadata = build_cache_metadata(source_args, "similarity")
            save_cache(path, torch.eye(100), metadata)

            stage2_args = self._args(60)
            stage2_args.git_commit = "stage2-fix-commit"
            similarity = load_explicit_similarity_cache(stage2_args, path)

            self.assertTrue(torch.equal(similarity, torch.eye(100)))
            self.assertEqual(stage2_args.similarity_hash, metadata["config_hash"])

    def test_explicit_similarity_rejects_protocol_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "similarity.pt")
            source_args = self._args(60)
            metadata = build_cache_metadata(source_args, "similarity")
            save_cache(path, torch.eye(100), metadata)

            mismatched = self._args(40)
            with self.assertRaises(RuntimeError):
                load_explicit_similarity_cache(mismatched, path)


if __name__ == "__main__":
    unittest.main()
