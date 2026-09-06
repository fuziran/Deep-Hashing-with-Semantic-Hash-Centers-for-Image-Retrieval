"""Fixed-center competition for the C1 experiment."""
import math

import torch
import torch.nn.functional as F


def center_competition(continuous_codes, centers, targets, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Competition temperature must be finite and positive")
    logits = F.normalize(continuous_codes, dim=1) @ F.normalize(
        centers.detach().to(continuous_codes), dim=1
    ).T
    return F.cross_entropy(logits / temperature, targets)
