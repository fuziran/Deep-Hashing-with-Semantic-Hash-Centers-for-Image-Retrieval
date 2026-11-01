"""Parameter-space diagnostics shared with the CIFAR SAM experiment."""
import math
import torch


def sampled_gradient_diagnostics(clean, weighted_rank, parameters):
    parameters = [p for p in parameters if p.requires_grad]
    a = torch.autograd.grad(clean, parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(weighted_rank, parameters, retain_graph=True, allow_unused=True)
    aa = sum(g.detach().double().square().sum().item() for g in a if g is not None)
    bb = sum(g.detach().double().square().sum().item() for g in b if g is not None)
    ab = sum((x.detach().double()*y.detach().double()).sum().item()
             for x, y in zip(a, b) if x is not None and y is not None)
    na, nb = math.sqrt(aa), math.sqrt(bb)
    return dict(sampled_clean_grad_norm=na, sampled_weighted_rank_grad_norm=nb,
                sampled_rank_clean_ratio=nb/na if na > 1e-12 else None,
                sampled_rank_clean_cosine=max(-1., min(1., ab/(na*nb))) if min(na, nb) > 1e-12 else None,
                sampled_rank_positive_zero_gradient=bool(weighted_rank.detach() > 0 and nb <= 1e-12))
