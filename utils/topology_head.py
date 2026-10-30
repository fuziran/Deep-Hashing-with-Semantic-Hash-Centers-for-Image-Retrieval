"""Five-epoch cached-feature head pilot with matched topology controls.

The alignment/diversity control is inspired by CroVCA (arXiv:2510.27584).
This is an adaptation to frozen NFD features and symmetric binary Hamming.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.semantic_mixup import frozen_batchnorm

VERSION = "i3-b1-topology-anchored-head-pilot-v1"
MODES = ("center_mlp", "cv_control", "global_topology", "nfd_topology")
EPOCHS, BATCH_SIZE = 5, 256
LEARNING_RATE, MIN_LR, WEIGHT_DECAY = 1e-3, 1e-6, 1e-2
RATE_WEIGHT, TOPOLOGY_WEIGHT = .1, .1


class SmallHashHead(nn.Module):
    def __init__(self, features, bits):
        super().__init__()
        self.hidden = nn.Linear(features, features)
        self.output = nn.Linear(features, bits)
        self.balance = nn.BatchNorm1d(bits)
        for layer in (self.hidden, self.output):
            nn.init.trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        x = F.normalize(features, dim=1)
        return self.balance(self.output(F.relu(self.hidden(x))))


def coding_rate_loss(logits):
    count, bits = logits.shape
    unit = F.normalize(logits, dim=1)
    matrix = torch.eye(bits, dtype=logits.dtype, device=logits.device)
    matrix = matrix + (bits / (count * .05)) * (unit.T @ unit)
    factor, status = torch.linalg.cholesky_ex(matrix)
    if (status != 0).any():
        raise FloatingPointError("Coding-rate matrix is not positive definite")
    return -factor.diagonal().log().sum() * ((bits + count) / (bits * count))


def class_targets(logits, labels, classes):
    with torch.no_grad():
        membership = F.one_hot(labels, classes).to(logits)
        totals = membership.T @ logits.detach()
        counts = membership.sum(0)[:, None].clamp_min(1)
        return ((totals / counts)[labels] >= 0).to(logits)


def topology_weights(neighbors, classes, local):
    if (neighbors.ndim != 2 or len(neighbors) != classes or neighbors.dtype != torch.long
            or (neighbors < 0).any() or (neighbors >= classes).any()
            or (neighbors == torch.arange(classes, device=neighbors.device)[:, None]).any()):
        raise ValueError("Invalid frozen NFD neighbors")
    offdiag = 1 - torch.eye(classes, device=neighbors.device)
    global_weight = offdiag / offdiag.sum()
    if not local:
        return global_weight
    adjacency = torch.zeros_like(offdiag)
    adjacency.scatter_(1, neighbors, 1)
    adjacency = torch.maximum(adjacency, adjacency.T) * offdiag
    if adjacency.sum() == 0:
        raise ValueError("Empty NFD neighbor graph")
    return .5 * global_weight + .5 * adjacency / adjacency.sum()


def topology_loss(prototype_logits, centers, weights):
    if (prototype_logits.ndim != 2 or prototype_logits.shape != centers.shape
            or weights.shape != (len(centers), len(centers))
            or not torch.isfinite(prototype_logits).all()
            or not ((centers == -1) | (centers == 1)).all()):
        raise ValueError("Invalid topology logits, targets or weights")
    soft_code = prototype_logits.tanh()
    predicted = (soft_code @ soft_code.T) / centers.shape[1]
    desired = (centers @ centers.T) / centers.shape[1]
    return ((predicted - desired).square() * weights).sum()


def fit_head(features, labels, centers, neighbors, mode, seed, device,
             epochs=EPOCHS, batch_size=BATCH_SIZE, progress=None):
    """Only training views/labels enter this function; no held-out input exists."""
    if mode not in MODES or epochs < 1 or batch_size < 2:
        raise ValueError("Invalid head pilot mode or budget")
    x = torch.as_tensor(features).detach().float()
    labels = torch.as_tensor(labels).detach().long()
    centers = torch.as_tensor(centers).detach().float()
    neighbors = torch.as_tensor(neighbors).detach().long()
    if (x.ndim != 3 or x.shape[1] != 2 or len(x) < 2 or labels.shape != (len(x),)
            or centers.ndim != 2 or not ((centers == -1) | (centers == 1)).all()
            or not torch.isfinite(x).all() or (x.norm(dim=2) <= 1e-12).any()
            or (labels < 0).any() or (labels >= len(centers)).any()
            or (torch.bincount(labels, minlength=len(centers)) == 0).any()):
        raise ValueError("Invalid training views or missing training class")
    x = F.normalize(x, dim=2).to(device)
    labels, centers, neighbors = labels.to(device), centers.to(device), neighbors.to(device)
    prototypes = torch.stack([x[labels == y].mean((0, 1)) for y in range(len(centers))])
    prototypes = F.normalize(prototypes, dim=1)
    graph = topology_weights(neighbors, len(centers), mode == "nfd_topology").to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = SmallHashHead(x.shape[2], centers.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed + 17)
    steps = math.ceil(len(x) / batch_size) * epochs
    history, global_step = [], 0
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(x), generator=generator)
        totals = dict(loss=0., alignment=0., rate=0., topology=0.)
        observed = 0
        for start in range(0, len(x), batch_size):
            ids = order[start:start + batch_size].to(device)
            current = len(ids)
            paired = torch.cat((x[ids, 0], x[ids, 1]), dim=0)
            targets = labels[ids].repeat(2)
            lr = MIN_LR + .5 * (LEARNING_RATE - MIN_LR) * (
                1 + math.cos(math.pi * global_step / max(steps - 1, 1)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = head(paired)
            rate = topology = logits.new_zeros(())
            if mode == "center_mlp":
                alignment = F.binary_cross_entropy_with_logits(2 * logits, (centers[targets] + 1) / 2)
                loss = alignment + 1e-4 * (logits.tanh().abs() - 1).square().mean()
            else:
                binary_targets = class_targets(logits, targets, len(centers))
                alignment = F.binary_cross_entropy_with_logits(logits, binary_targets)
                rate = .5 * (coding_rate_loss(logits[:current]) + coding_rate_loss(logits[current:]))
                loss = alignment + RATE_WEIGHT * rate
                if mode in ("global_topology", "nfd_topology"):
                    # Prototype-only pass receives gradients without changing BN running statistics.
                    with frozen_batchnorm(head):
                        prototype_logits = head(prototypes)
                    topology = topology_loss(prototype_logits, centers, graph)
                    loss = loss + TOPOLOGY_WEIGHT * topology
            if not torch.isfinite(loss):
                raise FloatingPointError("Head training loss is not finite")
            loss.backward()
            optimizer.step()
            for key, value in (("loss", loss), ("alignment", alignment),
                               ("rate", rate), ("topology", topology)):
                totals[key] += float(value.detach()) * current
            observed += current
            global_step += 1
        record = dict(epoch=epoch + 1, **{key: value / observed for key, value in totals.items()})
        history.append(record)
        if progress:
            progress(mode, record)
    head.eval()
    return head, history, dict(mode=mode, seed=seed, epochs=epochs, batch_size=batch_size,
                               steps=global_step, training_images=len(x), views_per_image=2,
                               parameters=sum(p.numel() for p in head.parameters()),
                               selected_epoch=epochs, selection="fixed final epoch; no validation selection")
