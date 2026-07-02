import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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

    for cand in candidates:
        if cand and os.path.exists(os.path.join(cand, "train")) and os.path.exists(os.path.join(cand, "test")):
            return cand

    raise FileNotFoundError(
        "Cannot find CIFAR-100 raw files. Please put files as: "
        "./data/cifar-100-python/train, ./data/cifar-100-python/test, ./data/cifar-100-python/meta "
        "or pass --root to the directory containing train/test/meta."
    )


def _load_cifar100_raw(root):
    root = _resolve_cifar100_root(root)

    train_obj = _unpickle(os.path.join(root, "train"))
    test_obj = _unpickle(os.path.join(root, "test"))

    train_data = train_obj["data"]
    test_data = test_obj["data"]

    train_labels = np.array(train_obj["fine_labels"], dtype=np.int64)
    test_labels = np.array(test_obj["fine_labels"], dtype=np.int64)

    data = np.concatenate([train_data, test_data], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    # CIFAR raw format: N x 3072, ordered as R(1024), G(1024), B(1024)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return images, labels


def _stratified_split_cifar100(labels, seed=60):
    """
    CIFAR-100 has 100 classes and 600 images per class.
    Paper setting:
      training: 10000 = 100/class
      query:     5000 =  50/class
      database: 45000 = 450/class
    """
    rng = np.random.RandomState(seed)

    train_indices = []
    query_indices = []
    database_indices = []

    for c in range(100):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)

        train_indices.extend(idx[:100])
        query_indices.extend(idx[100:150])
        database_indices.extend(idx[150:])

    train_indices = np.array(train_indices, dtype=np.int64)
    query_indices = np.array(query_indices, dtype=np.int64)
    database_indices = np.array(database_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(query_indices)
    rng.shuffle(database_indices)

    return train_indices, query_indices, database_indices


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
        img = Image.fromarray(self.images[real_idx])
        label_id = int(self.labels[real_idx])

        if self.transform is not None:
            img = self.transform(img)

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[label_id] = 1.0

        return img, label, torch.tensor(real_idx, dtype=torch.long)


def load_data(args):
    dataset_name = args.dataset.lower()

    if "cifar" not in dataset_name:
        raise NotImplementedError(
            "This patched data_loader currently supports CIFAR-100 only. "
            "For Stanford Cars/NABirds, please provide their directory/list-file structure first."
        )

    images, labels = _load_cifar100_raw(args.root)
    train_idx, query_idx, database_idx = _stratified_split_cifar100(labels, seed=args.seed)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CIFAR100HashDataset(
        images, labels, train_idx,
        num_classes=args.num_classes,
        transform=train_transform,
    )
    test_dataset = CIFAR100HashDataset(
        images, labels, query_idx,
        num_classes=args.num_classes,
        transform=test_transform,
    )
    database_dataset = CIFAR100HashDataset(
        images, labels, database_idx,
        num_classes=args.num_classes,
        transform=test_transform,
    )

    pin_memory = args.device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    database_loader = DataLoader(
        database_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print("========== CIFAR-100 dataset loaded ==========")
    print(f"train: {len(train_dataset)}, query: {len(test_dataset)}, database: {len(database_dataset)}")

    return (
        train_loader,
        test_loader,
        database_loader,
        len(train_dataset),
        len(test_dataset),
        len(database_dataset),
    )