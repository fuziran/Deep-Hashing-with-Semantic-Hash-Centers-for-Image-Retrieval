"""Copy only approved training auxiliary PNGs into a portable directory."""
import argparse
import csv
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-package", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.crop_package), Path(args.output)
    manifest = source / "metadata/applied_train_crops.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["index"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate manifest")
    image_root = (source / "images").resolve(strict=True)
    paths = []
    for row in rows:
        if row["split"] != "train" or row["visual_review"] != "accepted_crop":
            raise ValueError("Only approved training images may be packaged")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Image path must be relative without parent traversal")
        path = (image_root / relative).resolve(strict=True)
        if image_root not in path.parents:
            raise ValueError("Image path escapes root")
        paths.append(path)
    output.mkdir(parents=True, exist_ok=False)
    for row, path in zip(rows, paths):
        destination = output / "images" / Path(row["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(manifest, output / "approved_train.csv")
    print(f"Packaged {len(rows)} auxiliary PNGs at {output.resolve()}")
    print("Training membership, labels and pixel checksums are verified by run.py before use.")


if __name__ == "__main__":
    main()
