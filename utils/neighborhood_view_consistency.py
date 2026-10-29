"""NFD-neighborhood screened weak/strong binary-code consistency (I3 pilot).

The frozen NFD teacher decides whether a strong view preserves class-neighbor
directions. Only correct weak-view bits at risk on the strong view are aligned.
This is an exploratory, validation-only candidate, not a novelty claim.
"""
import math

import torch
from torch.nn import functional as F

VERSION = "i3-nfd-neighborhood-view-consistency-v1"


def ramp(epoch, maximum, warmup, ramp_end):
    if (epoch < 1 or not math.isfinite(maximum) or maximum < 0
            or not 0 <= warmup < ramp_end):
        raise ValueError("Invalid NVC weight schedule")
    return maximum * min(1., max(0., (epoch - warmup) / (ramp_end - warmup)))


@torch.no_grad()
def semantic_weight(weak_teacher, strong_teacher, basis, ranks, targets, temperature):
    if (weak_teacher.shape != strong_teacher.shape or weak_teacher.ndim != 2
            or basis.ndim != 3 or basis.shape[1] != weak_teacher.shape[1]
            or targets.shape != (len(weak_teacher),) or ranks.shape != (len(basis),)
            or not math.isfinite(temperature) or temperature <= 0
            or any(not torch.isfinite(x).all() for x in (weak_teacher, strong_teacher, basis))):
        raise ValueError("Invalid NVC teacher views, basis, or temperature")
    if ((targets < 0).any() or (targets >= len(basis)).any()
            or (weak_teacher.norm(dim=1) <= 1e-12).any()
            or (strong_teacher.norm(dim=1) <= 1e-12).any()):
        raise ValueError("Invalid NVC class or zero teacher feature")
    delta = F.normalize(weak_teacher, dim=1) - F.normalize(strong_teacher, dim=1)
    selected = basis[targets].to(delta)
    projected = torch.bmm(delta[:, None, :], selected).squeeze(1)
    active = (torch.arange(selected.shape[2], device=delta.device)[None, :]
              < ranks[targets].to(delta.device)[:, None])
    numerator = (projected.square() * active).sum(1)
    denominator = delta.square().sum(1)
    ratio = (numerator / denominator.clamp_min(1e-12)).clamp(0, 1)
    ratio = torch.where(denominator <= 1e-12, torch.zeros_like(ratio), ratio)
    return torch.exp(-ratio / temperature), ratio


def consistency_loss(weak_logits, strong_logits, targets, centers, gate,
                     margin=.5):
    if (weak_logits.shape != strong_logits.shape or weak_logits.ndim != 2
            or centers.ndim != 2 or centers.shape[1] != weak_logits.shape[1]
            or targets.shape != (len(weak_logits),) or gate.shape != (len(weak_logits),)
            or not math.isfinite(margin) or margin <= 0
            or any(not torch.isfinite(x).all() for x in (weak_logits, strong_logits, gate))
            or (gate < 0).any() or (gate > 1).any()
            or not ((centers == -1) | (centers == 1)).all()
            or (targets < 0).any() or (targets >= len(centers)).any()):
        raise ValueError("Invalid NVC logits, class centers, or gate")
    target = centers[targets].to(strong_logits)
    weak_correct = torch.where(weak_logits.detach() >= 0, 1., -1.) == target
    strong_margin = target * strong_logits
    risk = weak_correct & (strong_margin.detach() < margin)
    per_bit = F.softplus(margin - strong_margin)
    count = risk.sum()
    loss = (per_bit * risk * gate[:, None].detach()).sum() / count.clamp_min(1)
    return loss, dict(nvc_risk_bit_fraction=float(risk.float().mean()),
                      nvc_changed_bit_fraction=float((
                          (weak_logits.detach() >= 0) != (strong_logits.detach() >= 0)
                      ).float().mean()),
                      nvc_gate_mean=float(gate.detach().mean()),
                      nvc_active_bits=int(count))
