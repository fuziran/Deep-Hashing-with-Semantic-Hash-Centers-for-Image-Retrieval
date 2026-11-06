"""Dataset checks run without a PyTorch or GPU installation."""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from PIL import Image

from data.nabirds_manifest import read_verified_nabirds
from tools.prepare_nabirds_disjoint import audit, materialize, plan, verify


class TestNABirdsPreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original" / "images"
        number = 0
        for split, count in (("train", 4), ("query", 2), ("database", 2)):
            for class_name in ("001", "002"):
                folder = self.source / split / class_name
                folder.mkdir(parents=True)
                for index in range(count):
                    number += 1
                    image = Image.new("RGB", (9, 7), (number * 11 % 256, number * 31 % 256, number * 7 % 256))
                    image.save(folder / f"image_{number:03}.png")
        # Real source errors: training/query images copied into the database.
        for split in ("train", "query"):
            original = next((self.source / split / "001").glob("*.png"))
            shutil.copyfile(original, self.source / "database" / "001" / original.name)
        # Same pixels under a different ID and a different file encoding.
        query = next((self.source / "query" / "002").glob("*.png"))
        with Image.open(query) as image:
            image.save(self.source / "database" / "002" / "renamed_copy.bmp")
        self.args = argparse.Namespace(source=self.source, audit_dir=self.root / "audit",
                                       archive=None, workers=2, seed=60, val_per_class=2,
                                       train_pool_count=8, query_count=4,
                                       resume_incomplete=False,
                                       destination=self.root / "verified")

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            audit(self.args)
            plan(self.args)
            materialize(self.args)

    def test_full_audit_frozen_split_and_independent_copies(self):
        original_bytes = {str(p): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        self.prepare()
        root, metadata, rows = read_verified_nabirds(self.args.destination)
        self.assertEqual(metadata["counts"], {"train": 4, "validation": 4, "query": 4, "database": 4})
        self.assertEqual(metadata["removed_extra_copies"], 3)
        self.assertEqual(len({r["rgb_sha256"] for r in rows}), 16)
        self.assertEqual(original_bytes, {str(p): p.read_bytes() for p in self.source.rglob("*") if p.is_file()})
        self.assertEqual(read_verified_nabirds(root / "images")[1]["dataset_sha256"], metadata["dataset_sha256"])
        before = (self.args.audit_dir / "planned_images.jsonl").read_bytes()
        with redirect_stdout(io.StringIO()):
            plan(self.args)
        self.assertEqual(before, (self.args.audit_dir / "planned_images.jsonl").read_bytes())
        first = rows[0]
        target = root / first["path"]
        content = bytearray(target.read_bytes())
        content[-1] ^= 1
        target.write_bytes(content)
        self.assertEqual((self.source / first["source_path"]).read_bytes(), original_bytes[str(self.source / first["source_path"])])
        with self.assertRaisesRegex(ValueError, "content checksum"):
            read_verified_nabirds(root)

    def test_cross_label_content_conflict_aborts(self):
        original = next((self.source / "train" / "001").glob("*.png"))
        shutil.copyfile(original, self.source / "database" / "002" / "wrong_label.png")
        with redirect_stdout(io.StringIO()):
            audit(self.args)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                plan(self.args)
        self.assertFalse(self.args.destination.exists())

    def test_reused_identity_with_different_content_aborts(self):
        original = next((self.source / "train" / "001").glob("*.png"))
        Image.new("RGB", (9, 7), (250, 250, 250)).save(self.source / "query" / "001" / original.name)
        with redirect_stdout(io.StringIO()):
            audit(self.args)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                plan(self.args)

    def test_manifest_tampering_and_extra_files_are_rejected(self):
        self.prepare()
        path = self.args.destination / "manifests" / "query.jsonl"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            read_verified_nabirds(self.args.destination)
        path.write_bytes(original)
        source = next((self.args.destination / "images" / "query").rglob("*.png"))
        shutil.copyfile(source, source.with_name("extra.png"))
        with self.assertRaisesRegex(ValueError, "membership"):
            read_verified_nabirds(self.args.destination)

    def test_archive_verified_without_modification(self):
        archive = self.root / "original.tar.gz"
        with tarfile.open(archive, "w:gz") as packed:
            packed.add(self.source, arcname="NAbirds-new-seg/images")
        before = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.args.archive = archive
        with redirect_stdout(io.StringIO()):
            audit(self.args)
        result = json.loads((self.args.audit_dir / "archive_verification.json").read_text())
        self.assertTrue(result["matches_local_images"])
        self.assertEqual(result["sha256"], before)
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), before)
        self.assertEqual(result["image_members"], 19)
        with redirect_stdout(io.StringIO()):
            plan(self.args)
            materialize(self.args)
            verify(self.args)
        verification = json.loads((self.args.destination / "verification_report.json").read_text())
        self.assertTrue(verification["passed"])
        self.assertTrue(verification["original_extracted_images_unchanged"])

    def test_incomplete_copy_is_rejected(self):
        self.prepare()
        (self.args.destination / "INCOMPLETE").write_text("unfinished")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_verified_nabirds(self.args.destination)

    def test_repartition_restores_classes_with_no_exclusive_database_images(self):
        source = self.root / "all_overlap" / "images"
        for class_number, class_name in enumerate(("001", "002")):
            for split in ("train", "query", "database"):
                (source / split / class_name).mkdir(parents=True)
            for i in range(6):
                original_split = "train" if i < 4 else "query"
                path = source / original_split / class_name / f"{class_name}_{i}.png"
                Image.new("RGB", (9, 7), (10 + class_number * 90 + i * 10, 15, 25)).save(path)
                shutil.copyfile(path, source / "database" / class_name / path.name)
        self.args.source = source
        self.args.train_pool_count = 6
        self.args.query_count = 2
        self.prepare()
        _, metadata, rows = read_verified_nabirds(self.args.destination)
        summary = json.loads((self.args.audit_dir / "audit_summary.json").read_text())
        self.assertEqual(summary["naive_database_deduplication"]["classes_without_database_positives"], 2)
        self.assertEqual(metadata["counts"], {"train": 2, "validation": 4, "query": 2, "database": 4})
        for name in ("001", "002"):
            self.assertTrue(any(r["class_name"] == name and r["split"] == "database" for r in rows))

    def test_resume_checks_existing_files_and_preserves_interrupted_copy(self):
        with redirect_stdout(io.StringIO()):
            audit(self.args)
            plan(self.args)
        rows = [json.loads(line) for line in (self.args.audit_dir / "planned_images.jsonl").read_text().splitlines()]
        target = self.args.destination / rows[0]["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"interrupted copy")
        (self.args.destination / "INCOMPLETE").write_text("unfinished")
        self.args.resume_incomplete = True
        with redirect_stdout(io.StringIO()):
            materialize(self.args)
        self.assertEqual(len(read_verified_nabirds(self.args.destination)[2]), 16)
        partial = self.args.destination / "interrupted_copies" / (rows[0]["path"] + ".partial")
        self.assertEqual(partial.read_bytes(), b"interrupted copy")

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                         "PyTorch and torchvision are required for loader integration")
    def test_training_seed_does_not_change_frozen_membership(self):
        import torch
        from data.data_loader import _load_nabirds_data
        self.prepare()
        memberships = []
        for seed in (60, 61):
            args = argparse.Namespace(root=str(self.args.destination), dataset="NABirds-B", num_classes=2,
                                      val_per_class=2, seed=seed, output_dir=str(self.root / f"run_{seed}"),
                                      resize_size=8, crop_size=8, batch_size=2, num_workers=0,
                                      device=torch.device("cpu"))
            with redirect_stdout(io.StringIO()):
                loaders = _load_nabirds_data(args)
            self.assertEqual(loaders[-4:], (4, 4, 4, 4))
            memberships.append((args.split_hash, [loader.dataset.indices.tolist() for loader in loaders[:5]]))
            images, labels, _ = next(iter(loaders[0]))
            self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
            self.assertEqual(tuple(labels.shape), (2, 2))
        self.assertEqual(memberships[0], memberships[1])

    def test_corrupt_image_prevents_split(self):
        (self.source / "train" / "001" / "broken.jpg").write_bytes(b"not an image")
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                audit(self.args)
            with self.assertRaisesRegex(ValueError, "decode errors"):
                plan(self.args)


if __name__ == "__main__":
    unittest.main()
