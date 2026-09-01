import hashlib
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _unpickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _resolve_cifar100_root(root):
    candidates = [
        root,
        os.path.join(root, "cifar-100-python"),
        "./data/cifar-100-python",
        "./data/cifar/cifar-100-python",
        "./datasets/CIFAR-100/cifar-100-python",
        "./datasets/cifar-100-python",
    ]
    for candidate in candidates:
        if candidate and all(
            os.path.exists(os.path.join(candidate, name))
            for name in ("train", "test", "meta")
        ):
            return candidate
    raise FileNotFoundError(
        "Cannot find CIFAR-100 raw files. Put train/test/meta under "
        "./data/cifar-100-python or pass --root explicitly."
    )


def _load_cifar100_raw(root):
    root = _resolve_cifar100_root(root)
    train_obj = _unpickle(os.path.join(root, "train"))
    test_obj = _unpickle(os.path.join(root, "test"))
    data = np.concatenate([train_obj["data"], test_obj["data"]], axis=0)
    labels = np.asarray(
        train_obj["fine_labels"] + test_obj["fine_labels"], dtype=np.int64
    )
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def _stratified_split_cifar100(labels, seed=60, val_per_class=10):
    """Create a fixed B0 split and keep the query set final-evaluation only."""
    if not 0 < val_per_class < 100:
        raise ValueError("val_per_class must be between 1 and 99")

    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []
    query_indices, database_indices = [], []
    for class_id in range(100):
        indices = np.where(labels == class_id)[0]
        if len(indices) != 600:
            raise ValueError(
                f"CIFAR-100 class {class_id} has {len(indices)} samples, expected 600"
            )
        rng.shuffle(indices)
        train_end = 100 - val_per_class
        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:100])
        query_indices.extend(indices[100:150])
        database_indices.extend(indices[150:])

    arrays = [
        np.asarray(values, dtype=np.int64)
        for values in (train_indices, val_indices, query_indices, database_indices)
    ]
    for values in arrays:
        rng.shuffle(values)
    return tuple(arrays)


def _split_sha256(*index_arrays):
    digest = hashlib.sha256()
    for indices in index_arrays:
        digest.update(np.asarray(indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def _save_split_manifest(args, labels, train_idx, val_idx, query_idx, database_idx):
    split_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    split_hash = _split_sha256(train_idx, val_idx, query_idx, database_idx)
    stem = f"{args.dataset}_seed{args.seed}_val{args.val_per_class}_{split_hash[:12]}"
    np.savez_compressed(
        os.path.join(split_dir, f"{stem}.npz"),
        train=train_idx,
        validation=val_idx,
        query=query_idx,
        database=database_idx,
        labels=np.asarray(labels, dtype=np.int64),
    )
    manifest = {
        "dataset": args.dataset,
        "seed": args.seed,
        "validation_per_class": args.val_per_class,
        "counts": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "query": int(len(query_idx)),
            "database": int(len(database_idx)),
        },
        "sha256": split_hash,
    }
    with open(os.path.join(split_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return split_hash


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


class CIFAR100HashDataset(Dataset):
    def __init__(self, images, labels, indices, num_classes=100, transform=None):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        real_idx = int(self.indices[item])
        image = Image.fromarray(self.images[real_idx])
        if self.transform is not None:
            image = self.transform(image)
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[int(self.labels[real_idx])] = 1.0
        return image, label, torch.tensor(real_idx, dtype=torch.long)


def load_data(args):
    if "cifar" not in args.dataset.lower():
        raise NotImplementedError(
            "The B0 loader currently supports CIFAR-100 only. "
            "Cars/NABirds need a separately audited manifest loader."
        )

    images, labels = _load_cifar100_raw(args.root)
    train_idx, val_idx, query_idx, database_idx = _stratified_split_cifar100(
        labels, seed=args.seed, val_per_class=args.val_per_class
    )
    args.split_hash = _save_split_manifest(
        args, labels, train_idx, val_idx, query_idx, database_idx
    )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(args.resize_size),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(args.resize_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    def make_dataset(indices, transform):
        return CIFAR100HashDataset(
            images, labels, indices, args.num_classes, transform=transform
        )

    train_dataset = make_dataset(train_idx, train_transform)
    relation_dataset = make_dataset(train_idx, eval_transform)
    val_dataset = make_dataset(val_idx, eval_transform)
    query_dataset = make_dataset(query_idx, eval_transform)
    database_dataset = make_dataset(database_idx, eval_transform)

    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.type == "cuda",
        "drop_last": False,
        "worker_init_fn": _seed_worker,
    }
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common_loader_args
    )
    relation_loader = DataLoader(relation_dataset, shuffle=False, **common_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
    query_loader = DataLoader(query_dataset, shuffle=False, **common_loader_args)
    database_loader = DataLoader(database_dataset, shuffle=False, **common_loader_args)

    print("========== CIFAR-100 dataset loaded ==========")
    print(
        f"train: {len(train_dataset)}, validation: {len(val_dataset)}, "
        f"query: {len(query_dataset)}, database: {len(database_dataset)}"
    )
    print(f"split sha256: {args.split_hash}")
    return (
        train_loader,
        relation_loader,
        val_loader,
        query_loader,
        database_loader,
        len(train_dataset),
        len(val_dataset),
        len(query_dataset),
        len(database_dataset),
    )
