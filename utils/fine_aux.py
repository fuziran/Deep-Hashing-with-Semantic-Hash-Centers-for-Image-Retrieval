"""Training-only fine-label classification on shared student features."""
import torch
from torch import nn
from torch.nn import functional as F

from utils.neighbor_feature_distill import nfd_weight

VERSION = "fine-aux-v1"


def fine_aux_weight(epoch, weight, warmup, ramp_end):
    return nfd_weight(epoch, weight, warmup, ramp_end)


def build_fine_aux(feature_dim, num_classes, seed, device):
    # Isolate head initialization from student/data augmentation RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 300007)
        head = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(head.weight, 0, .01)
        nn.init.zeros_(head.bias)
    return head.to(device)


def fine_aux_loss(head, features, labels):
    if (labels.ndim != 2 or labels.shape != (features.shape[0], head.out_features)
            or not torch.all((labels == 0) | (labels == 1))
            or not torch.all(labels.sum(1) == 1)):
        raise ValueError("Fine auxiliary classification requires one-hot single-class labels")
    logits = head(features)
    targets = labels.argmax(1)
    return F.cross_entropy(logits, targets), (logits.argmax(1) == targets).float().mean()
