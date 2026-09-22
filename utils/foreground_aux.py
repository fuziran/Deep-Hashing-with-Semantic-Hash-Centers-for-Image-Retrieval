"""Audited, train-only auxiliary PNGs; never replace dataset images."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def pixel_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_auxiliary(manifest, root, images, labels, train_indices, transform):
    root = Path(root).resolve(strict=True)
    allowed = set(map(int, train_indices))
    records, tensors, originals, auxiliary = [], {}, {}, {}
    with open(manifest, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Auxiliary manifest is empty")
    for row in rows:
        idx, target = int(row["index"]), int(row["class_id"])
        if idx not in allowed or row["split"] != "train":
            raise ValueError(f"Auxiliary index {idx} is not in the current training split")
        if idx in tensors:
            raise ValueError(f"Duplicate auxiliary index: {idx}")
        if target != int(labels[idx]):
            raise ValueError(f"Auxiliary label mismatch: {idx}")
        if row["visual_review"] != "accepted_crop":
            raise ValueError(f"Auxiliary image is not approved: {idx}")
        if pixel_digest(images[idx]) != row["pixel_sha256"]:
            raise ValueError(f"Original pixel mismatch at {idx}; --root must use original images")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Auxiliary path must be relative without parent traversal")
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("Auxiliary path escapes image root")
        with Image.open(path) as im:
            if im.mode != "RGB" or im.size != (32, 32):
                raise ValueError(f"Expected RGB 32x32 auxiliary image: {path}")
            array = np.array(im)
        digest = pixel_digest(array)
        if digest != row["output_pixel_sha256"]:
            raise ValueError(f"Auxiliary pixel checksum mismatch: {idx}")
        original = Image.fromarray(images[idx])
        aux = Image.fromarray(array)
        tensors[idx] = transform(aux)
        originals[idx], auxiliary[idx] = original, aux
        records.append({"index": idx, "class_id": target,
                        "original_sha256": row["pixel_sha256"],
                        "auxiliary_sha256": digest})
    records.sort(key=lambda r: r["index"])
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"records": records, "tensors": tensors, "originals": originals,
            "images": auxiliary, "sha256": digest, "transform": transform}


def save_previews(bundle, directory):
    directory = Path(directory) / "aux_previews"
    directory.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    def rendered(tensor):
        pixels = ((tensor.cpu() * std + mean).clamp(0, 1) * 255).round().byte()
        return Image.fromarray(pixels.permute(1, 2, 0).numpy())
    for row in bundle["records"]:
        idx = row["index"]
        views = [bundle["originals"][idx], bundle["images"][idx],
                 rendered(bundle["transform"](bundle["originals"][idx])),
                 rendered(bundle["tensors"][idx])]
        panel = Image.new("RGB", (4 * 224, 250), "white")
        draw = ImageDraw.Draw(panel)
        for col, (view, title) in enumerate(zip(views, ["original PNG", "auxiliary PNG", "original model input", "auxiliary model input"])):
            panel.paste(view.resize((224, 224)), (col * 224, 26))
            draw.text((col * 224 + 4, 5), title, fill="black")
        panel.save(directory / f"{idx:05d}_class{row['class_id']}.png")


def export_diagnostics(artifact, directory):
    if "auxiliary_diagnostics" not in artifact:
        return
    directory = Path(directory)
    (directory / "auxiliary_diagnostics.json").write_text(
        json.dumps(artifact["auxiliary_diagnostics"], indent=2), encoding="utf-8")
    for key in ("S_prob_original", "S_prob", "S1_original", "S1"):
        np.savetxt(directory / f"{key}.csv", artifact[key].cpu().numpy(), delimiter=",")
