"""Read and verify the frozen NABirds-B dataset, without PyTorch dependencies."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

SPLITS = ("train", "validation", "query", "database")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_verified_nabirds(root, verify_files=True):
    """Return (root, metadata, rows), or None for an unverified legacy root.

    Frozen manifests are independent of the training random seed. File hashes
    bind the earlier pixel audit to the exact bytes consumed by training.
    """
    root = Path(root).resolve()
    if root.name == "images" and (root.parent / "dataset_manifest.json").is_file():
        root = root.parent
    manifest_path = root / "dataset_manifest.json"
    if (root / "INCOMPLETE").exists():
        raise ValueError("Verified NABirds copy is incomplete; do not train")
    if not manifest_path.is_file():
        if (root / "images" / "validation").exists() or (root / "validation").exists():
            raise ValueError("Validation directory found without a frozen dataset_manifest.json")
        return None
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("protocol") != "NABirds-B-verified-disjoint-v1":
        raise ValueError("Unsupported verified NABirds dataset protocol")
    classes = metadata["class_names"]
    if not classes or classes != sorted(set(classes)):
        raise ValueError("Class names must be sorted and unique")
    expected_manifests = {f"manifests/{split}.jsonl" for split in SPLITS}
    manifest_hashes = metadata["manifests"]
    if set(manifest_hashes) != expected_manifests:
        raise ValueError("Expected exactly four fixed split manifests")
    bundle_hash = hashlib.sha256(json.dumps(manifest_hashes, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    if bundle_hash != metadata["dataset_sha256"]:
        raise ValueError("Dataset manifest bundle checksum mismatch")
    seen = {key: set() for key in ("path", "image_id", "sha256", "rgb_sha256", "display_rgb_sha256")}
    rows = []
    for split in SPLITS:
        filename = f"manifests/{split}.jsonl"
        if file_sha256(root / filename) != manifest_hashes[filename]:
            raise ValueError(f"Split manifest checksum mismatch: {split}")
        with (root / filename).open(encoding="utf-8") as stream:
            subset = [json.loads(line) for line in stream if line.strip()]
        if len(subset) != metadata["counts"][split]:
            raise ValueError(f"Manifest count mismatch: {split}")
        counts = Counter()
        for row in subset:
            relative = PurePosixPath(row["path"])
            if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 4 or
                    "\\" in row["path"] or relative.parts[:2] != ("images", split) or
                    row["split"] != split or relative.parts[2] != row["class_name"] or
                    relative.suffix.lower() not in EXTENSIONS):
                raise ValueError(f"Invalid image path or split: {row['path']}")
            if (not isinstance(row["class_index"], int) or
                    not 0 <= row["class_index"] < len(classes) or
                    classes[row["class_index"]] != row["class_name"]):
                raise ValueError(f"Class index mismatch: {row['path']}")
            if row["image_id"] != relative.stem.casefold():
                raise ValueError(f"Image ID mismatch: {row['path']}")
            for key in seen:
                value = row[key].casefold() if key == "path" else row[key]
                if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError(f"Invalid {key}: {row['path']}")
                if value in seen[key]:
                    raise ValueError(f"Duplicate {key} within or across frozen splits: {row['path']}")
                seen[key].add(value)
            path = root / row["path"]
            if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing or unsafe image file: {path}")
            if path.stat().st_size != row["size"]:
                raise ValueError(f"Image size mismatch: {path}")
            counts[row["class_name"]] += 1
        if set(counts) != set(classes):
            raise ValueError(f"Every class must occur in {split}")
        if split == "validation" and set(counts.values()) != {metadata["validation_per_class"]}:
            raise ValueError("Frozen validation-per-class mismatch")
        rows.extend(subset)
    actual_paths = {p.relative_to(root).as_posix().casefold() for p in (root / "images").rglob("*")
                    if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if actual_paths != seen["path"]:
        raise ValueError("Image directory membership differs from the frozen manifests")
    if verify_files:
        def check(row):
            if file_sha256(root / row["path"]) != row["sha256"]:
                raise ValueError(f"Image content checksum mismatch: {row['path']}")
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(check, rows):
                pass
    return root, metadata, rows
