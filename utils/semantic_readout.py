"""Fixed-budget linear readout fit on frozen NFD student features."""
import torch
from torch.nn import functional as F

VERSION = "i3-nfd-prototype-difference-readout-screen-v1"
RIDGE_MULTIPLIER = 1.0
GEOMETRY_WEIGHT = .1


@torch.no_grad()
def fit_readout(features, labels, centers, prototypes, neighbors, geometry="none"):
    """Fit normalized-feature regression with optional prototype-difference rows.

    Only training features/labels and frozen parent geometry may be supplied.
    There is no validation-based coefficient or hyperparameter selection.
    """
    if geometry not in ("none", "global", "nfd"):
        raise ValueError("Unknown readout geometry")
    x = torch.as_tensor(features).detach().cpu().double()
    y = torch.as_tensor(labels).detach().cpu().long()
    h = torch.as_tensor(centers).detach().cpu().double()
    p = torch.as_tensor(prototypes).detach().cpu().double()
    near = torch.as_tensor(neighbors).detach().cpu()
    if (x.ndim != 2 or h.ndim != 2 or p.ndim != 2 or len(x) < 2
            or y.shape != (len(x),) or len(h) != len(p) or p.shape[1] != x.shape[1]
            or near.ndim != 2 or len(near) != len(h) or near.dtype != torch.long
            or near.shape[1] < 1 or (y < 0).any() or (y >= len(h)).any()
            or (near < 0).any() or (near >= len(h)).any()
            or not ((h == -1) | (h == 1)).all()
            or not all(torch.isfinite(t).all() for t in (x, h, p))
            or (x.norm(dim=1) <= 1e-12).any() or (p.norm(dim=1) <= 1e-12).any()
            or (near == torch.arange(len(h))[:, None]).any()
            or any(len(set(row.tolist())) != len(row) for row in near)):
        raise ValueError("Invalid training features, center targets or frozen geometry")
    x, p = F.normalize(x, dim=1), F.normalize(p, dim=1)
    targets = h[y]
    mean_x, mean_target = x.mean(0), targets.mean(0)
    centered_x, centered_target = x - mean_x, targets - mean_target
    gram = centered_x.T @ centered_x / len(x)
    cross = centered_x.T @ centered_target / len(x)
    ridge = RIDGE_MULTIPLIER * gram.trace() / x.shape[1]
    if ridge <= 1e-12:
        raise ValueError("Training features have no usable variance")
    edge_features = edge_targets = None
    edge_count = 0
    if geometry != "none":
        if geometry == "nfd":
            anchors = torch.arange(len(h))[:, None].expand_as(near).reshape(-1)
            rivals = near.reshape(-1)
        else:
            anchors = torch.arange(len(h)).repeat_interleave(len(h))
            rivals = torch.arange(len(h)).repeat(len(h))
            keep = anchors != rivals
            anchors, rivals = anchors[keep], rivals[keep]
        edge_features = p[anchors] - p[rivals]
        edge_targets = h[anchors] - h[rivals]
        edge_count = len(edge_features)
        # Match total input-Gram trace across global and local graph controls.
        edge_scale = edge_features.square().sum(1).mean()
        if edge_scale <= 1e-12:
            raise ValueError("Prototype differences have no usable variation")
        denominator = edge_count * edge_scale
        gram = gram + GEOMETRY_WEIGHT * (edge_features.T @ edge_features) / denominator
        cross = cross + GEOMETRY_WEIGHT * (edge_features.T @ edge_targets) / denominator
    weight = torch.linalg.solve(gram + ridge * torch.eye(x.shape[1], dtype=x.dtype), cross)
    bias = mean_target - mean_x @ weight
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise FloatingPointError("Readout solution is not finite")
    metadata = dict(version=VERSION, geometry=geometry, training_count=len(x),
                    feature_dim=x.shape[1], code_length=h.shape[1], edge_count=edge_count,
                    feature_policy="L2-normalized frozen student features",
                    ridge_multiplier=RIDGE_MULTIPLIER, ridge_lambda=float(ridge),
                    geometry_weight=GEOMETRY_WEIGHT if geometry != "none" else 0.,
                    training_center_mse=float((x @ weight + bias - targets).square().mean()),
                    prototype_difference_mse=(float((edge_features @ weight - edge_targets)
                                                     .square().mean())
                                              if edge_features is not None else None))
    return dict(weight=weight, bias=bias, metadata=metadata)


@torch.no_grad()
def readout_logits(features, head):
    x = torch.as_tensor(features).detach().cpu().double()
    weight, bias = head["weight"], head["bias"]
    if (x.ndim != 2 or x.shape[1] != weight.shape[0]
            or bias.shape != (weight.shape[1],) or not torch.isfinite(x).all()
            or (x.norm(dim=1) <= 1e-12).any()):
        raise ValueError("Invalid readout features or head")
    return F.normalize(x, dim=1) @ weight + bias
