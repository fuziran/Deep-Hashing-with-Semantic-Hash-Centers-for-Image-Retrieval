"""CPU-only regressions. No training, dataset download, or pretrained weights."""
import copy
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tests.test_i1_relation import args_for, ToyClassifier, toy_loader
from utils.foreground_aux import load_auxiliary, pixel_digest, export_diagnostics, save_previews
from utils.reliable_relation import estimate_relations, view_statistics
from utils.experiment import build_cache_metadata, save_cache
from GenerateSimilarityMatrix import TrainClassificationNetwork, normalize_and_symmetrize_similarity
from data.data_loader import _resolve_cifar100_root
from network import _build_resnet


class AuxLoader(list):
    def __init__(self, batches, auxiliary):
        super().__init__(batches)
        self.dataset = SimpleNamespace(auxiliary=auxiliary)


class TestAuxiliaryRelations(unittest.TestCase):
    def test_zero_exact_and_positive_probability_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            original = estimate_relations(ToyClassifier(), toy_loader(), args)
            bundle = {"sha256": "a" * 64, "tensors": {1: torch.tensor([[[2., 0.], [1., 3.]]])}}
            loader = AuxLoader(toy_loader(), bundle)
            args.aux_manifest, args.aux_content_sha256 = "approved.csv", bundle["sha256"]
            args.aux_count, args.aux_alpha = 1, 0.0
            zero = estimate_relations(ToyClassifier(), loader, args)
            for key in ("similarity", "S_prob", "S_feat", "e_c", "gate", "prototypes"):
                self.assertTrue(torch.equal(original[key], zero[key]), key)
            args.aux_alpha = .2
            changed = estimate_relations(ToyClassifier(), loader, args)
            for key in ("S0", "S_two_view", "S_feat", "e_c", "gate", "prototypes"):
                self.assertTrue(torch.equal(original[key], changed[key]), key)
            self.assertFalse(torch.equal(original["S_prob"], changed["S_prob"]))
            # Independent weighted aggregation: exactly one global index gets mixed.
            images, labels, indices = [torch.cat([b[k] for b in toy_loader()]) for k in range(3)]
            net = ToyClassifier()
            z1, _ = net.forward_with_features(images)
            z2, _ = net.forward_with_features(images.flip(-1))
            targets = labels.argmax(1)
            stats = view_statistics(z1, z2, targets)
            rows = (stats["a1"] + stats["a2"]) / 2
            za, _ = net.forward_with_features(bundle["tensors"][1][None])
            za = za.clone()
            za[0, za.argmax(1).item()] = -float("inf")
            rows[1] = .8 * rows[1] + .2 * za.softmax(1)[0]
            expected = torch.stack([(rows[targets == c] * stats["weight"][targets == c, None]).sum(0)
                                    / stats["weight"][targets == c].sum() for c in range(3)])
            torch.testing.assert_close(changed["S_prob"], normalize_and_symmetrize_similarity(expected))
            self.assertEqual(changed["auxiliary_diagnostics"]["class_aux_count"], [0, 1, 0])
            export_diagnostics(changed, directory)
            self.assertTrue((Path(directory) / "auxiliary_diagnostics.json").exists())
            combined = AuxLoader([(images, labels, indices)], bundle)
            torch.testing.assert_close(changed["similarity"], estimate_relations(net, combined, args)["similarity"])
            bundle["tensors"][99] = bundle["tensors"][1]
            with self.assertRaisesRegex(ValueError, "every declared"):
                estimate_relations(net, loader, args)

    def test_auxiliary_cache_changes_but_classifier_shared(self):
        args = args_for("unused", classifier_sha256="f" * 64, data_sha256="original")
        base_classifier = build_cache_metadata(args, "classifier")
        base_relation = build_cache_metadata(args, "similarity")
        args.aux_manifest, args.aux_alpha = "approved.csv", .2
        args.aux_content_sha256, args.aux_count = "a" * 64, 56
        self.assertEqual(base_classifier, build_cache_metadata(args, "classifier"))
        first = build_cache_metadata(args, "similarity")
        self.assertNotEqual(base_relation, first)
        for key, value in (("aux_alpha", .1), ("aux_content_sha256", "b" * 64), ("data_sha256", "changed")):
            changed = copy.copy(args)
            setattr(changed, key, value)
            self.assertNotEqual(first, build_cache_metadata(changed, "similarity"))


class TestAuxiliaryManifest(unittest.TestCase):
    def test_audited_images_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
            aux = np.full((32, 32, 3), 120, dtype=np.uint8)
            Image.fromarray(aux).save(root / "one.png")
            row = dict(index=0, class_id=1, split="train", visual_review="accepted_crop",
                       relative_path="one.png", pixel_sha256=pixel_digest(images[0]),
                       output_pixel_sha256=pixel_digest(aux))
            manifest = root / "approved.csv"
            def write(rows):
                with manifest.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                [.485, .456, .406], [.229, .224, .225])])
            def load():
                return load_auxiliary(manifest, root, images, np.array([1, 0]), [0], transform)
            write([row])
            bundle = load()
            save_previews(bundle, directory)
            self.assertTrue((root / "aux_previews/00000_class1.png").exists())
            self.assertEqual(len(bundle["tensors"]), 1)
            write([row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load()
            for key, value in (("index", 1), ("class_id", 2), ("split", "query"),
                               ("pixel_sha256", "wrong"), ("output_pixel_sha256", "wrong"),
                               ("visual_review", "rejected_crop"), ("relative_path", "../one.png")):
                bad = dict(row, **{key: value})
                write([bad])
                with self.assertRaises(ValueError):
                    load()
            write([dict(row, relative_path="missing.png")])
            with self.assertRaises(FileNotFoundError):
                load()
            write([row])
            images[0, 0, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, "Original pixel"):
                load()

    def test_invalid_root_never_falls_back(self):
        with self.assertRaises(FileNotFoundError):
            _resolve_cifar100_root("this-root-does-not-exist-aux-test")


class TestClassifierReuse(unittest.TestCase):
    def test_cache_backbone_construction_avoids_pretrained_weights(self):
        with patch.dict("network.resnet_dict", {"ResNet34": lambda **kwargs: kwargs}):
            self.assertEqual(_build_resnet("ResNet34", pretrained=False), {"weights": None})

    def test_diagnostics_never_train_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory, relation_diagnostics_only=True, data_sha256="original")
            with patch("GenerateSimilarityMatrix.ClassifyNet", side_effect=AssertionError("training attempted")):
                with self.assertRaisesRegex(RuntimeError, "training is disabled"):
                    TrainClassificationNetwork(args, None, None)
            path = Path(directory) / "classifier.pt"
            metadata = build_cache_metadata(args, "classifier")
            save_cache(path, ToyClassifier().state_dict(), metadata)
            args.classifier_cache = str(path)
            args.classifier_cache_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch("GenerateSimilarityMatrix.ClassifyNet", return_value=ToyClassifier()):
                self.assertIsInstance(TrainClassificationNetwork(args, None, None), ToyClassifier)
                args.data_sha256 = "cut-images"
                with self.assertRaisesRegex(RuntimeError, "Incompatible"):
                    TrainClassificationNetwork(args, None, None)
            args.data_sha256 = "original"
            payload = torch.load(path, map_location="cpu")
            payload["state_sha256"] = "wrong"
            torch.save(payload, path)
            args.classifier_cache_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "state SHA256"):
                TrainClassificationNetwork(args, None, None)
            del args.data_sha256
            save_cache(path, {}, build_cache_metadata(args, "classifier"))
            args.classifier_cache_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "Legacy"):
                TrainClassificationNetwork(args, None, None)


if __name__ == "__main__":
    unittest.main()
