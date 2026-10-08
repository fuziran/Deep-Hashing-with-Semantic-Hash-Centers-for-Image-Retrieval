import argparse
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from data.data_loader import _load_nabirds_data, _stratified_split_nabirds_train


class TestNABirdsSplit(unittest.TestCase):
    def test_stratified_holdout_is_disjoint_and_seeded(self):
        labels = np.repeat(np.arange(3), 4)
        first = _stratified_split_nabirds_train(labels, 3, seed=60, val_per_class=2)
        second = _stratified_split_nabirds_train(labels, 3, seed=60, val_per_class=2)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        train, validation = first
        self.assertEqual(len(set(train) & set(validation)), 0)
        np.testing.assert_array_equal(np.bincount(labels[train]), [2, 2, 2])
        np.testing.assert_array_equal(np.bincount(labels[validation]), [2, 2, 2])

    def test_too_small_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "class 1"):
            _stratified_split_nabirds_train(np.array([0, 0, 0, 1, 1]), 2, 60, 2)

    def test_loader_keeps_curated_splits_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.join(directory, "images")
            for split, count in (("train", 4), ("query", 2), ("database", 2)):
                for class_name in ("bird_a", "bird_b", "bird_c"):
                    folder = os.path.join(root, split, class_name)
                    os.makedirs(folder)
                    for index in range(count):
                        Image.new("RGB", (8, 8)).save(
                            os.path.join(folder, f"{index}.jpg")
                        )
            args = argparse.Namespace(
                root=directory, dataset="NABirds-B", num_classes=3,
                val_per_class=2, seed=60, output_dir=os.path.join(directory, "out"),
                resize_size=8, crop_size=8, batch_size=2, num_workers=0,
                device=torch.device("cpu"),
            )
            result = _load_nabirds_data(args)
            self.assertEqual(result[-4:], (6, 6, 6, 6))
            train, relation, validation, query, database = result[:5]
            self.assertEqual(train.dataset.indices.tolist(), relation.dataset.indices.tolist())
            self.assertTrue(set(train.dataset.indices).isdisjoint(validation.dataset.indices))
            self.assertTrue(set(query.dataset.indices).isdisjoint(database.dataset.indices))
            self.assertEqual(query.dataset[0][1].sum().item(), 1)
            split_json = os.path.join(args.output_dir, "splits", os.listdir(os.path.join(args.output_dir, "splits"))[0])
            if not split_json.endswith(".json"):
                split_json = split_json[:-4] + ".json"
            with open(split_json, encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["class_names"], ["bird_a", "bird_b", "bird_c"])
            self.assertEqual(manifest["sha256"], args.split_hash)
            self.assertEqual(manifest["counts"], {
                "train": 6, "validation": 6, "query": 6, "database": 6,
            })


if __name__ == "__main__":
    unittest.main()
