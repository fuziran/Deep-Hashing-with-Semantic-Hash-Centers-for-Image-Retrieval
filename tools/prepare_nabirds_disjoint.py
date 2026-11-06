"""Audit the supplied NABirds-B archive and build a separate, frozen dataset.

Never rewrites source images. Requires Pillow and NumPy; no training dependencies.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import tarfile
import threading

import numpy as np
import PIL
from PIL import Image, ImageOps

SPLITS = ("train", "query", "database")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_hash(path):
    with path.open("rb") as stream:
        return stream_hash(stream)


def stream_hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def pixel_hash(image):
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def inspect_image(item):
    root, path = item
    relative = path.relative_to(root).as_posix()
    parts = PurePosixPath(relative).parts
    row = {"source_path": relative, "source_split": parts[0], "class_name": parts[1],
           "image_id": path.stem.casefold()}
    try:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source image must be a regular file inside the source root")
        before = path.stat()
        data = path.read_bytes()
        row.update(size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                   source_mtime_ns=before.st_mtime_ns)
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            row.update(width=image.width, height=image.height, mode=image.mode,
                       rgb_sha256=pixel_hash(image),
                       display_rgb_sha256=pixel_hash(ImageOps.exif_transpose(image)))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed while reading")
        row["error"] = None
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def summarize(rows):
    good = [row for row in rows if not row["error"]]
    result = {"source_counts": dict(Counter(row["source_split"] for row in rows)),
              "files_checked": len(rows), "decode_errors": sum(bool(r["error"]) for r in rows),
              "classes": len({r["class_name"] for r in rows}), "duplicate_checks": {}}
    for field in ("image_id", "sha256", "rgb_sha256", "display_rgb_sha256"):
        groups = defaultdict(list)
        for row in good:
            groups[row[field]].append(row)
        duplicate_groups = [group for group in groups.values() if len(group) > 1]
        pairs = Counter()
        for group in duplicate_groups:
            for i, left in enumerate(group):
                for right in group[i + 1:]:
                    pairs["/".join(sorted((left["source_split"], right["source_split"])))] += 1
        result["duplicate_checks"][field] = {
            "unique": len(groups), "groups": len(duplicate_groups),
            "excess_copies": sum(len(g) - 1 for g in duplicate_groups), "pairs": dict(pairs),
            "cross_label_groups": sum(len({r["class_name"] for r in g}) > 1 for g in groups.values()),
        }
    membership = defaultdict(lambda: defaultdict(set))
    for row in good:
        membership[row["class_name"]][row["source_split"]].add(row["image_id"])
    empty_classes = sorted(name for name, group in membership.items()
                           if not (group["database"] - group["train"] - group["query"]))
    result["naive_database_deduplication"] = {
        "classes_without_database_positives": len(empty_classes), "class_names": empty_classes,
        "consequence": "Deleting database copies alone cannot yield a valid all-class retrieval protocol."
                       if empty_classes else "All classes retain database positives.",
    }
    return result


class HashingReader:
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.raw.read(size)
        self.digest.update(data)
        return data


def check_archive(archive, source_rows, output):
    local = {r["source_path"]: r for r in source_rows}
    seen = set()
    mismatches, unrecognized, duplicate_members = [], [], []
    archive_rows = []
    before = archive.stat()
    with archive.open("rb") as raw:
        reader = HashingReader(raw)
        with tarfile.open(fileobj=reader, mode="r|gz") as packed:
            for member in packed:
                if not member.isfile() or PurePosixPath(member.name).suffix.lower() not in EXTENSIONS:
                    continue
                parts = PurePosixPath(member.name).parts
                candidates = [i for i, name in enumerate(parts[:-2])
                              if name == "images" and parts[i + 1] in SPLITS]
                if len(candidates) != 1:
                    unrecognized.append(member.name)
                    continue
                relative = PurePosixPath(*parts[candidates[0] + 1:]).as_posix()
                content = packed.extractfile(member)
                digest = stream_hash(content)
                archive_rows.append({"member": member.name, "source_path": relative,
                                     "sha256": digest, "size": member.size})
                if relative in seen:
                    duplicate_members.append(relative)
                seen.add(relative)
                if relative not in local or local[relative].get("sha256") != digest:
                    mismatches.append(relative)
                if len(archive_rows) % 5000 == 0:
                    print(f"Archive images checked: {len(archive_rows)}", flush=True)
        while reader.read(8 * 1024 * 1024):
            pass
        sha256 = reader.digest.hexdigest()
    after = archive.stat()
    result = {"path": str(archive.resolve()), "size": before.st_size,
              "mtime_ns": before.st_mtime_ns, "sha256": sha256,
              "archive_unchanged_during_audit": (before.st_size, before.st_mtime_ns) ==
                                                   (after.st_size, after.st_mtime_ns),
              "image_members": len(archive_rows), "mismatched_images": mismatches,
              "local_images_absent_from_archive": sorted(set(local) - seen),
              "unrecognized_image_members": unrecognized, "duplicate_member_paths": duplicate_members}
    result["matches_local_images"] = not any((mismatches, result["local_images_absent_from_archive"],
                                              unrecognized, duplicate_members))
    write_rows(output / "archive_inventory.jsonl", archive_rows)
    write_json(output / "archive_verification.json", result)
    return result


def audit(args):
    root, output = args.source.resolve(), args.audit_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    images = [p for p in files if p.suffix.lower() in EXTENSIONS]
    unexpected = [str(p.relative_to(root)) for p in files if p.suffix.lower() not in EXTENSIONS]
    if not images or any(p.relative_to(root).parts[0] not in SPLITS or
                         len(p.relative_to(root).parts) < 3 for p in images):
        raise ValueError("Expected source root with train/query/database class directories")
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, row in enumerate(pool.map(inspect_image, ((root, p) for p in images)), 1):
            rows.append(row)
            if count % 2000 == 0:
                print(f"Local images decoded and hashed: {count}/{len(images)}", flush=True)
    write_rows(output / "source_inventory.jsonl", rows)
    result = summarize(rows)
    result.update(source_root=str(root), non_image_files=unexpected,
                  software_versions={"python": platform.python_version(), "numpy": np.__version__,
                                     "pillow": PIL.__version__,
                                     "libjpeg": getattr(Image.core, "jpeglib_version", None)},
                  pixel_hash_format="sha256(b'RGB:{width}x{height}:' + RGB pixels)",
                  scope="All image IDs, file bytes, decoded RGB pixels and EXIF-oriented RGB pixels; "
                        "not an exhaustive visual near-duplicate or same-bird identity analysis.")
    write_json(output / "audit_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if args.archive:
        archive_result = check_archive(args.archive.resolve(), rows, output)
        print(json.dumps(archive_result, indent=2), flush=True)
        if not archive_result["matches_local_images"] or not archive_result["archive_unchanged_during_audit"]:
            raise ValueError("Archive/local mismatch: inspect archive_verification.json")
    if result["decode_errors"]:
        raise ValueError("Corrupt or unstable images found: inspect inventory before planning")


def allocate_counts(sizes, target, minimum, maximum):
    """Constrained proportional integer quotas, exact total and stable tie breaks."""
    total = sum(sizes.values())
    if not sum(minimum.values()) <= target <= sum(maximum.values()):
        raise ValueError("Requested split size cannot satisfy per-class coverage constraints")
    allocation = {name: min(max(sizes[name] * target // total, minimum[name]), maximum[name])
                  for name in sizes}
    while sum(allocation.values()) < target:
        name = min((n for n in sizes if allocation[n] < maximum[n]),
                   key=lambda n: (-(sizes[n] * target - allocation[n] * total), n))
        allocation[name] += 1
    while sum(allocation.values()) > target:
        name = min((n for n in sizes if allocation[n] > minimum[n]),
                   key=lambda n: (sizes[n] * target - allocation[n] * total, n))
        allocation[name] -= 1
    return allocation


def plan(args):
    output = args.audit_dir.resolve()
    rows = read_rows(output / "source_inventory.jsonl")
    if any(r["error"] for r in rows):
        raise ValueError("Cannot split images with unresolved decode errors")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for field in ("image_id", "sha256", "rgb_sha256", "display_rgb_sha256"):
        seen = {}
        for i, row in enumerate(rows):
            if row[field] in seen:
                parents[find(i)] = find(seen[row[field]])
            else:
                seen[row[field]] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    selected, removed, conflicts = [], [], []
    priority = {"query": 0, "train": 1, "database": 2}
    for group in groups.values():
        if len({r["class_name"] for r in group}) > 1:
            conflicts.append({"reason": "same identity/content across labels", "images": group})
            continue
        # Equal IDs with different displayed pixels require explicit investigation.
        by_id = defaultdict(set)
        for row in group:
            by_id[row["image_id"]].add(row["display_rgb_sha256"])
        if any(len(values) > 1 for values in by_id.values()):
            conflicts.append({"reason": "same image ID with differing pixels", "images": group})
            continue
        ordered = sorted(group, key=lambda r: (priority[r["source_split"]], r["source_path"]))
        kept = dict(ordered[0])
        kept["split"] = kept["source_split"]
        selected.append(kept)
        for row in ordered[1:]:
            removed.append({"removed_source_path": row["source_path"],
                            "kept_source_path": kept["source_path"],
                            "shared_checks": [f for f in ("image_id", "sha256", "rgb_sha256",
                                                          "display_rgb_sha256") if row[f] == kept[f]]})
    write_json(output / "conflicts.json", conflicts)
    if conflicts:
        raise ValueError("Identity/label conflicts found; inspect conflicts.json before choosing a policy")
    classes = sorted({r["class_name"] for r in selected})
    sizes = Counter(r["class_name"] for r in selected)
    training_counts = allocate_counts(sizes, args.train_pool_count,
                                      {c: args.val_per_class + 1 for c in classes},
                                      {c: sizes[c] - 2 for c in classes})
    query_counts = allocate_counts(sizes, args.query_count, {c: 1 for c in classes},
                                   {c: sizes[c] - training_counts[c] - 1 for c in classes})
    rng = np.random.RandomState(args.seed)
    for class_name in classes:
        class_rows = sorted([r for r in selected if r["class_name"] == class_name],
                            key=lambda r: (r["image_id"], r["sha256"]))
        indices = np.arange(len(class_rows))
        rng.shuffle(indices)
        train_end = training_counts[class_name]
        query_end = train_end + query_counts[class_name]
        for position, index in enumerate(indices):
            if position < args.val_per_class:
                split = "validation"
            elif position < train_end:
                split = "train"
            elif position < query_end:
                split = "query"
            else:
                split = "database"
            class_rows[int(index)]["split"] = split
    for row in selected:
        row["class_index"] = classes.index(row["class_name"])
        row["path"] = f"images/{row['split']}/{row['class_name']}/{PurePosixPath(row['source_path']).name}"
    selected.sort(key=lambda r: (r["split"], r["class_name"], r["path"]))
    if len({r["path"].casefold() for r in selected}) != len(selected):
        raise ValueError("Output filenames collide")
    write_rows(output / "planned_images.jsonl", selected)
    write_rows(output / "removed_duplicates.jsonl", sorted(removed, key=lambda r: r["removed_source_path"]))
    per_class = {name: dict(Counter(r["split"] for r in selected if r["class_name"] == name)) for name in classes}
    result = {"protocol": "NABirds-B-verified-disjoint-v1", "seed": args.seed,
              "validation_per_class": args.val_per_class,
              "counts": dict(Counter(r["split"] for r in selected)),
              "unique_images": len(selected), "removed_extra_copies": len(removed),
              "class_names": classes, "per_class_counts": per_class,
              "training_pool_count": args.train_pool_count, "query_target_count": args.query_count,
              "policy": "One representative per connected image-ID/byte/RGB/display-RGB group, "
                        "canonical copy priority query > train > database, then lexicographic source path. "
                        "Repartition ALL unique images by class with constrained proportional integer quotas "
                        "and exact global training-pool/query sizes. Guarantee at least one training, query "
                        "and database image per class plus fixed validation_per_class. Within each sorted "
                        "class sort by image ID/content hash, shuffle once using NumPy RandomState(seed), "
                        "then take validation, train, query, database in that order. Original memberships "
                        "are not preserved. Conflicting labels or same-ID differing pixels abort. "
                        "This is not the author's recovered split.",
              "plan_sha256": file_hash(output / "planned_images.jsonl")}
    write_json(output / "split_plan.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_class_counts", "class_names")}, indent=2))


def materialize(args):
    audit_dir, destination = args.audit_dir.resolve(), args.destination.resolve()
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    source = Path(summary["source_root"]).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Destination must be separate from source")
    if destination.exists() and not args.resume_incomplete:
        raise FileExistsError(f"Will not overwrite existing directory: {destination}")
    plan_info = json.loads((audit_dir / "split_plan.json").read_text(encoding="utf-8"))
    if file_hash(audit_dir / "planned_images.jsonl") != plan_info["plan_sha256"]:
        raise ValueError("Plan checksum mismatch")
    rows = read_rows(audit_dir / "planned_images.jsonl")
    marker = destination / "INCOMPLETE"
    if destination.exists():
        if not marker.is_file() or (destination / "dataset_manifest.json").exists():
            raise ValueError("Resume is allowed only for an unfinished copy")
        expected = {r["path"] for r in rows}
        present = {p.relative_to(destination).as_posix() for p in (destination / "images").rglob("*")
                   if p.is_file() and p.suffix.lower() in EXTENSIONS}
        if not present <= expected:
            raise ValueError("Unplanned images in unfinished destination")
    else:
        destination.mkdir(parents=True)
    marker.write_text("Copy in progress; do not train.\n", encoding="utf-8")
    manifests = destination / "manifests"
    manifests.mkdir(exist_ok=True)
    image_folders = set()
    for row in rows:
        for field in ("source_path", "path"):
            relative = PurePosixPath(row[field])
            if relative.is_absolute() or ".." in relative.parts or "\\" in row[field]:
                raise ValueError(f"Unsafe planned relative path: {row[field]}")
        image_folders.add((destination / row["path"]).parent)
    # Create shared parent directories before parallel I/O. Concurrent recursive
    # mkdir can fail with transient Access Denied errors on Windows.
    for folder in sorted(image_folders):
        folder.mkdir(parents=True, exist_ok=True)
        if not folder.resolve().is_relative_to(destination):
            raise ValueError(f"Image directory escapes dataset root: {folder}")
    interrupted_directory_lock = threading.Lock()

    def copy_image(row):
        original, target = source / row["source_path"], destination / row["path"]
        if (not original.resolve().is_relative_to(source) or
                not target.parent.resolve().is_relative_to(destination) or target.is_symlink()):
            raise ValueError(f"Path escapes dataset root: {original.resolve()} / {source}; "
                             f"{target.parent.resolve()} / {destination}")
        if target.exists():
            if file_hash(target) == row["sha256"]:
                return
            # Preserve a copy interrupted mid-write before trying the image again.
            interrupted = destination / "interrupted_copies" / (row["path"] + ".partial")
            if not interrupted.resolve().is_relative_to(destination) or interrupted.exists():
                raise ValueError("Unsafe interrupted-copy destination")
            with interrupted_directory_lock:
                interrupted.parent.mkdir(parents=True, exist_ok=True)
            target.replace(interrupted)
        # Independent copies keep subsequent edits isolated from the preserved source.
        shutil.copy2(original, target)
        if file_hash(target) != row["sha256"]:
            raise ValueError(f"Copied file failed checksum: {target}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, _ in enumerate(pool.map(copy_image, rows), 1):
            if count % 3000 == 0:
                print(f"Copied and verified: {count}/{len(rows)}", flush=True)
    manifest_hashes = {}
    fields = ("path", "split", "class_name", "class_index", "image_id", "sha256", "rgb_sha256",
              "display_rgb_sha256", "size", "source_path")
    for split in ("train", "validation", "query", "database"):
        subset = [{key: row[key] for key in fields} for row in rows if row["split"] == split]
        filename = f"manifests/{split}.jsonl"
        write_rows(destination / filename, subset)
        manifest_hashes[filename] = file_hash(destination / filename)
    provenance = destination / "audit"
    shutil.copytree(audit_dir, provenance, dirs_exist_ok=args.resume_incomplete)
    protocol = {k: v for k, v in plan_info.items() if k != "per_class_counts"}
    protocol.update(schema_version=1, source_root=str(source), manifests=manifest_hashes,
                    dataset_sha256=hashlib.sha256(json.dumps(manifest_hashes, sort_keys=True,
                                                           separators=(",", ":")).encode()).hexdigest(),
                    audit_hashes={p.name: file_hash(p) for p in sorted(provenance.iterdir()) if p.is_file()})
    write_json(destination / "dataset_manifest.json", protocol)
    (destination / "README.txt").write_text(
        "NABirds-B verified disjoint v1\n"
        "Original archive and extracted data remain unchanged. Images here are independent byte-identical copies.\n"
        "Four fixed splits: images/train, images/validation, images/query, images/database.\n"
        "Use the dataset ROOT with run.py --dataset NABirds-B --root <this folder>.\n"
        "The updated loader consumes manifests and does not resample validation when --seed changes.\n"
        "This is a verified disjoint protocol, not a recovery of the author's original split.\n"
        "Audit covers file IDs, exact bytes and exact decoded/oriented pixels; resized/recompressed/cropped\n"
        "near-duplicates and different photos of the same individual bird are not ruled out.\n"
        f"Counts: {json.dumps(protocol['counts'], sort_keys=True)}\n"
        f"Dataset SHA256: {protocol['dataset_sha256']}\n", encoding="utf-8")
    (destination / "INCOMPLETE").unlink()
    print(json.dumps({k: v for k, v in protocol.items() if k != "class_names"}, indent=2), flush=True)


def verify(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.nabirds_manifest import read_verified_nabirds
    print("Verifying frozen manifests and every destination image SHA256...", flush=True)
    root, metadata, rows = read_verified_nabirds(args.destination)
    print(f"Destination images verified: {len(rows)}. Rechecking preserved source images...", flush=True)
    source_rows = read_rows(root / "audit" / "source_inventory.jsonl")
    source = Path(metadata["source_root"])

    def source_unchanged(row):
        path = source / row["source_path"]
        stat = path.stat()
        return (stat.st_size == row["size"] and stat.st_mtime_ns == row["source_mtime_ns"]
                and file_hash(path) == row["sha256"])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        unchanged = all(list(pool.map(source_unchanged, source_rows)))
    current_source_paths = {p.relative_to(source).as_posix() for p in source.rglob("*")
                            if p.is_file() and p.suffix.lower() in EXTENSIONS}
    unchanged = unchanged and current_source_paths == {row["source_path"] for row in source_rows}
    print(f"Preserved source images checked: {len(source_rows)}. Rechecking original archive SHA256...", flush=True)
    archive = json.loads((root / "audit" / "archive_verification.json").read_text(encoding="utf-8"))
    archive_path = Path(archive["path"])
    archive_stat = archive_path.stat()
    archive_unchanged = (archive_stat.st_size == archive["size"] and
                         archive_stat.st_mtime_ns == archive["mtime_ns"] and
                         file_hash(archive_path) == archive["sha256"])
    print("Archive recheck complete. Writing pairwise-intersection and class-coverage report...", flush=True)
    intersections = {}
    for field in ("image_id", "sha256", "rgb_sha256", "display_rgb_sha256"):
        values = {split: {r[field] for r in rows if r["split"] == split}
                  for split in ("train", "validation", "query", "database")}
        intersections[field] = {f"{a}/{b}": len(values[a] & values[b])
                                for a in values for b in values if a < b}
    per_class = {name: dict(Counter(r["split"] for r in rows if r["class_name"] == name))
                 for name in metadata["class_names"]}
    result = {"passed": unchanged and archive_unchanged,
              "protocol": metadata["protocol"], "counts": metadata["counts"],
              "classes": len(metadata["class_names"]), "dataset_sha256": metadata["dataset_sha256"],
              "all_destination_image_bytes_verified": True,
              "original_extracted_images_unchanged": unchanged,
              "original_archive_unchanged": archive_unchanged,
              "original_archive_sha256": archive["sha256"],
              "pairwise_intersections": intersections, "per_class_counts": per_class}
    write_json(root / "verification_report.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_class_counts"}, indent=2), flush=True)
    if not result["passed"]:
        raise ValueError("Verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("audit", "plan", "materialize", "verify"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--val-per-class", type=int, default=2)
    parser.add_argument("--train-pool-count", type=int, default=16384)
    parser.add_argument("--query-count", type=int, default=8171)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume-incomplete", action="store_true",
                        help="Resume only a destination containing INCOMPLETE; verify existing copies first")
    args = parser.parse_args()
    if args.val_per_class < 1:
        parser.error("--val-per-class must be positive")
    if args.operation == "audit" and not args.source:
        parser.error("audit requires --source")
    if args.operation in ("materialize", "verify") and not args.destination:
        parser.error("materialize/verify require --destination")
    globals()[args.operation](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
