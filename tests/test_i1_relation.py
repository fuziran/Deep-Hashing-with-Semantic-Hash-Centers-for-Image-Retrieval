"""Small CPU tests; no image dataset, pretrained download, or training required."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from GenerateSimilarityMatrix import (
    GenerateSimilarityMatrix, load_explicit_similarity_cache,
    normalize_and_symmetrize_similarity,
)
from network import ClassifyNet
from utils.experiment import build_cache_metadata, save_cache
from utils.reliable_relation import (
    MODES, estimate_relations, fuse_relations, generate_i1, relation_config,
    state_sha256, tensor_sha256, validate_similarity, view_statistics,
)


def args_for(directory, **overrides):
    values = dict(
        variant="b1", relation_mode="full", relation_kappa=30.0,
        relation_temperature=1.0, relation_weight_floor=0.05,
        num_classes=3, device=torch.device("cpu"), resize_size=4, crop_size=4,
        mask_strategy="predicted_argmax", dataset="toy", seed=60,
        split_hash="train-only", git_commit="test", lr=7e-5,
        classify_epoch=300, code_length=4, force_recompute=False,
        similarity_cache=None, output_dir=directory, run_dir=directory,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ToyClassifier(nn.Module):
    def forward_with_features(self, images):
        features = images.flatten(1).double() + 1.0
        logits = features[:, :3] * torch.tensor([1., 2., 3.])
        return logits, features

    def forward(self, images):
        logits, _ = self.forward_with_features(images)
        return logits, logits.softmax(1)


def toy_loader():
    images = torch.arange(24, dtype=torch.float32).reshape(6, 1, 2, 2) / 12
    labels = torch.eye(3)[torch.tensor([0, 1, 1, 2, 2, 2])]
    return [(images[:2], labels[:2], torch.arange(2)),
            (images[2:], labels[2:], torch.arange(2, 6))]


class TestReliability(unittest.TestCase):
    def test_identical_views_have_unit_consistency_and_true_class_weight(self):
        z = torch.tensor([[2., 1., 0.], [0., 0., 0.]], dtype=torch.float64)
        targets = torch.tensor([0, 1])
        result = view_statistics(z, z, targets)
        torch.testing.assert_close(result["consistency"], torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(result["weight"], z.softmax(1)[torch.arange(2), targets])

    def test_disjoint_extreme_predictions_floor_and_independent_masks(self):
        z1 = torch.tensor([[10000., -10000., -10000.]])
        z2 = torch.tensor([[-10000., 10000., -10000.]])
        result = view_statistics(z1, z2, torch.tensor([0]))
        self.assertAlmostEqual(result["weight"].item(), 0.05)
        self.assertAlmostEqual(result["js"].item(), __import__("math").log(2))
        self.assertEqual(result["a1"][0, 0].item(), 0)
        self.assertEqual(result["a2"][0, 1].item(), 0)
        for value in result.values():
            self.assertTrue(torch.isfinite(value).all())
        truth = view_statistics(z1, z2, torch.tensor([2]), mask_strategy="ground_truth")
        self.assertEqual(truth["a1"][0, 2].item(), 0)
        self.assertEqual(truth["a2"][0, 2].item(), 0)

    def test_nan_and_invalid_parameters_rejected(self):
        z = torch.zeros(1, 3)
        for temp in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                view_statistics(z, z, torch.tensor([0]), temperature=temp)
        with self.assertRaises(FloatingPointError):
            view_statistics(z + float("nan"), z, torch.tensor([0]))

    def test_gate_does_not_get_renormalized(self):
        probability = torch.tensor([[1., 0.8], [0.8, 1.]], dtype=torch.float64)
        feature = torch.tensor([[1., -0.4], [-0.4, 1.]], dtype=torch.float64)
        fused, gate = fuse_relations(probability, feature, torch.tensor([10., 10.]), 30.)
        self.assertAlmostEqual(gate[0, 1].item(), 0.25)
        self.assertAlmostEqual(fused[0, 1].item(), -0.1)
        validate_similarity(fused, 2)


class TestRelations(unittest.TestCase):
    def test_feature_api_preserves_original_forward(self):
        # Use the real class with a tiny backbone, avoiding pretrained loading.
        net = ClassifyNet.__new__(ClassifyNet)
        nn.Module.__init__(net)
        net.feature_layers = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        net.classify_layer = nn.Linear(4, 3)
        net.softmax = nn.Softmax(dim=1)
        x = torch.randn(2, 4, 2, 2)
        original_logits, probabilities = net(x)
        logits, features = net.forward_with_features(x)
        torch.testing.assert_close(original_logits, logits)
        torch.testing.assert_close(probabilities, logits.softmax(1))
        torch.testing.assert_close(features, x.mean((2, 3)))
        self.assertEqual(features.shape, (2, 4))

    def test_all_modes_batch_invariance_and_baseline_relation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            net = ToyClassifier()
            loader = toy_loader()
            artifact = estimate_relations(net, loader, args)
            combined = [tuple(torch.cat([batch[k] for batch in loader]) for k in range(3))]
            second = estimate_relations(net, combined, args)
            for key in ("S0", "S_prob", "S_feat", "S1", "prototypes", "e_c"):
                torch.testing.assert_close(artifact[key], second[key])
            torch.testing.assert_close(artifact["class_count"], torch.tensor([1., 2., 3.], dtype=torch.float64))
            # Independent sample-by-sample baseline probability aggregation.
            images, labels, _ = combined[0]
            z, f = net.forward_with_features(images)
            z[torch.arange(len(z)), z.argmax(1)] = float("-inf")
            p = z.softmax(1)
            averages = torch.stack([p[labels.argmax(1) == c].mean(0) for c in range(3)])
            torch.testing.assert_close(artifact["S0"], normalize_and_symmetrize_similarity(averages))
            self.assertEqual(artifact["prototypes"].shape, (3, 4))
            # Independently reconstruct weighted rows and equal-weight prototypes.
            z1, f1 = net.forward_with_features(images)
            z2, f2 = net.forward_with_features(images.flip(-1))
            targets = labels.argmax(1)
            weights, masked_means = [], []
            for i, target in enumerate(targets):
                p1, p2 = z1[i].softmax(0), z2[i].softmax(0)
                mid = (p1 + p2) / 2
                js = ((p1 * (p1.log() - mid.log())).sum()
                      + (p2 * (p2.log() - mid.log())).sum()) / 2
                weight = max(0.05, (1 - js.item() / __import__("math").log(2))
                             * (p1[target] + p2[target]).item() / 2)
                weights.append(weight)
                views = []
                for z_view in (z1[i].clone(), z2[i].clone()):
                    z_view[z_view.argmax()] = float("-inf")
                    views.append(z_view.softmax(0))
                masked_means.append((views[0] + views[1]) / 2)
            weights = torch.tensor(weights, dtype=torch.float64)
            masked_means = torch.stack(masked_means)
            rows = torch.stack([
                (masked_means[targets == c] * weights[targets == c, None]).sum(0)
                / weights[targets == c].sum() for c in range(3)
            ])
            torch.testing.assert_close(artifact["S_prob"], normalize_and_symmetrize_similarity(rows))
            sample_features = (f1 / f1.norm(dim=1, keepdim=True)
                               + f2 / f2.norm(dim=1, keepdim=True)) / 2
            prototypes = torch.stack([sample_features[targets == c].mean(0) for c in range(3)])
            prototypes /= prototypes.norm(dim=1, keepdim=True)
            torch.testing.assert_close(artifact["prototypes"], prototypes)
            expected = dict(full=artifact["S1"], two_view=artifact["S_two_view"],
                            weighted=artifact["S_prob"], feature_only=artifact["S_feat"],
                            fixed_fusion=(artifact["S_prob"] + artifact["S_feat"]) / 2)
            for mode in MODES:
                args.relation_mode = mode
                result = estimate_relations(net, loader, args)["similarity"]
                validate_similarity(result, 3)
                torch.testing.assert_close(result, expected[mode].float())

    def test_missing_classes_and_bad_features_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            with self.assertRaisesRegex(ValueError, "No Stage 1"):
                estimate_relations(ToyClassifier(), toy_loader()[:1], args)
            with self.assertRaises(ValueError):
                estimate_relations(ToyClassifier(), [], args)
            net = ToyClassifier()
            with patch.object(net, "forward_with_features", return_value=(torch.zeros(2, 3), torch.zeros(2, 4))):
                with self.assertRaisesRegex(ValueError, "Zero visual"):
                    estimate_relations(net, toy_loader(), args)

    def test_b0_does_not_enter_i1_and_matches_original_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory, variant="b0", force_recompute=True)
            net = ToyClassifier()
            expected = estimate_relations(net, toy_loader(), args)["S0"].float()
            with patch("GenerateSimilarityMatrix.TrainClassificationNetwork", return_value=net), \
                 patch("utils.reliable_relation.generate_i1", side_effect=AssertionError("I1 entered")):
                actual = GenerateSimilarityMatrix(args, None, toy_loader(), None)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestProvenance(unittest.TestCase):
    def test_cache_dependencies_and_classifier_sharing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory, classifier_sha256="a" * 64,
                            similarity_hash="config", similarity_content_sha256="b" * 64)
            baseline = copy.copy(args)
            baseline.variant = "b0"
            self.assertEqual(build_cache_metadata(args, "classifier"),
                             build_cache_metadata(baseline, "classifier"))
            reference = build_cache_metadata(args, "similarity")
            for field, value in (
                ("relation_kappa", 10.), ("relation_temperature", 2.),
                ("relation_mode", "weighted"), ("relation_weight_floor", 0.1),
                ("classifier_sha256", "c" * 64), ("split_hash", "different"),
                ("mask_strategy", "ground_truth"), ("crop_size", 3),
            ):
                changed = copy.copy(args)
                setattr(changed, field, value)
                self.assertNotEqual(reference, build_cache_metadata(changed, "similarity"))
            before = build_cache_metadata(args, "centers")
            args.similarity_content_sha256 = "c" * 64
            self.assertNotEqual(before, build_cache_metadata(args, "centers"))

    def test_roundtrip_hit_explicit_reuse_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            net = ToyClassifier()
            with patch("GenerateSimilarityMatrix.TrainClassificationNetwork", return_value=net):
                first = generate_i1(args, None, toy_loader(), None)
                with patch("utils.reliable_relation.estimate_relations", side_effect=AssertionError("cache miss")):
                    second = generate_i1(args, None, None, None)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            path = next((Path(directory) / "SimilarityMatrix").glob("*.pt"))
            args.git_commit = "later-stage-only-revision"
            restored = load_explicit_similarity_cache(args, path)
            torch.testing.assert_close(first, restored)
            with self.assertRaises(RuntimeError):
                load_explicit_similarity_cache(args_for(directory, variant="b0"), path)
            with self.assertRaises(RuntimeError):
                load_explicit_similarity_cache(args_for(directory, relation_kappa=10.), path)
            payload = torch.load(path, map_location="cpu")
            payload["value"]["similarity"][0, 1] = 0.123
            payload["value"]["similarity"][1, 0] = 0.123
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                load_explicit_similarity_cache(args, path)

    def test_tensor_and_state_content_hashes(self):
        tensor = torch.tensor([1., 2.])
        self.assertEqual(tensor_sha256(tensor), tensor_sha256(tensor.clone()))
        self.assertNotEqual(tensor_sha256(tensor), tensor_sha256(tensor + 1))
        self.assertNotEqual(state_sha256({"a": tensor}), state_sha256({"a": tensor + 1}))


if __name__ == "__main__":
    unittest.main()