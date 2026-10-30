"""Label-free two-view fusion from the NFD-aligned student feature space."""
import torch
from torch.nn import functional as F


VERSION = "i3-nfd-neighbor-view-fusion-screen-v1"


def bounded_view_weight(first_margin, second_margin):
    """Weight the more reliable view, never more than three to one."""
    if (first_margin.shape != second_margin.shape or first_margin.ndim != 1
            or not torch.isfinite(first_margin).all()
            or not torch.isfinite(second_margin).all()):
        raise ValueError("Invalid paired view margins")
    first = first_margin.clamp_min(0)
    second = second_margin.clamp_min(0)
    return .5 + .25 * (first - second) / (first + second + 1e-6)


def fused_logits(logits, flipped_logits, features, flipped_features,
                 prototypes, neighbors):
    """Return equal, all-rival, and frozen-NFD-neighbor fusion controls.

    The predicted anchor class comes from the two student features jointly;
    no true label, validation statistic or teacher forward is used.
    """
    if (logits.ndim != 2 or flipped_logits.shape != logits.shape
            or features.ndim != 2 or flipped_features.shape != features.shape
            or len(features) != len(logits) or prototypes.ndim != 2
            or features.shape[1] != prototypes.shape[1]
            or neighbors.ndim != 2 or neighbors.shape[0] != len(prototypes)
            or neighbors.shape[1] < 1 or not all(torch.isfinite(t).all() for t in
                                                 (logits, flipped_logits, features,
                                                  flipped_features, prototypes))
            or (features.norm(dim=1) <= 1e-12).any()
            or (flipped_features.norm(dim=1) <= 1e-12).any()
            or (prototypes.norm(dim=1) <= 1e-12).any()
            or neighbors.dtype != torch.long or (neighbors < 0).any()
            or (neighbors >= len(prototypes)).any()):
        raise ValueError("Invalid NFD view-fusion inputs")
    first = F.normalize(features, dim=1)
    second = F.normalize(flipped_features, dim=1)
    centers = F.normalize(prototypes.to(first), dim=1)
    anchor = (F.normalize(first + second, dim=1) @ centers.T).argmax(1)
    nearby = neighbors.to(anchor.device)[anchor]
    if (nearby == anchor[:, None]).any():
        raise ValueError("NFD neighbor list contains its anchor class")

    scores_first, scores_second = first @ centers.T, second @ centers.T
    rows = torch.arange(len(anchor), device=anchor.device)

    def margins(scores):
        own = scores[rows, anchor]
        all_rivals = scores.scatter(1, anchor[:, None], -torch.inf)
        global_margin = own - all_rivals.max(1).values
        neighbor_margin = own - scores.gather(1, nearby).max(1).values
        return global_margin, neighbor_margin

    global_first, neighbor_first = margins(scores_first)
    global_second, neighbor_second = margins(scores_second)
    global_weight = bounded_view_weight(global_first, global_second)
    neighbor_weight = bounded_view_weight(neighbor_first, neighbor_second)

    def combine(weight):
        return weight[:, None] * logits + (1 - weight[:, None]) * flipped_logits

    return dict(
        uniform=(logits + flipped_logits) / 2,
        global_prototype=combine(global_weight),
        nfd_neighbor=combine(neighbor_weight),
        global_weight=global_weight,
        neighbor_weight=neighbor_weight,
    )
