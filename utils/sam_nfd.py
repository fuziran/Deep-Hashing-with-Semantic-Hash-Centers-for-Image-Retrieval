"""Non-adaptive L2 SAM on the unchanged NFD objective (Foret et al., ICLR 2021).

Independent implementation. RMSprop remains the base optimizer. The second pass
uses the same inputs, frozen teacher features and random draws. Only the first
student pass updates BatchNorm running buffers. This is an existing-method
control, not a new hashing method.
"""
from contextlib import contextmanager
import math

import torch

VERSION = "i3-nfd-sam-control-v1"


@contextmanager
def second_pass_context(net, before_rng, after_rng, device):
    # Training BN still uses batch statistics; eval() would change the objective.
    states = []
    for module in net.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            buffers = {name: value.clone() for name, value in module.named_buffers(recurse=False)
                       if value is not None}
            states.append((module, module.momentum, buffers))
            module.momentum = 0.
    torch.set_rng_state(before_rng[0])
    if device.type == "cuda":
        torch.cuda.set_rng_state(before_rng[1], device)
    try:
        yield
    finally:
        with torch.no_grad():
            for module, momentum, buffers in states:
                module.momentum = momentum
                for name, value in buffers.items():
                    getattr(module, name).copy_(value)
        torch.set_rng_state(after_rng[0])
        if device.type == "cuda":
            torch.cuda.set_rng_state(after_rng[1], device)


def rng_state(device):
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def sam_nfd_step(net, optimizer, criterion, nfd, images, labels, indices,
                 args, strength, sample_gradients=False):
    """Two backward passes, one optimizer update; restore parameters on failure."""
    rho = args.sam_rho
    if not math.isfinite(rho) or rho <= 0 or nfd.gate_mode is not None:
        raise ValueError("SAM requires positive finite rho and ordinary fixed NFD")
    device = images.device
    before_rng = rng_state(device)
    optimizer.zero_grad(set_to_none=True)
    output, features = net.forward_with_features(images)
    clean = criterion(output, labels, indices, args)
    combined = clean
    teacher_features = None
    diagnostics = {}
    if strength > 0:
        weighted = strength * nfd(features, images, labels.argmax(1))
        teacher_features = nfd.last_teacher_features
        if sample_gradients:
            from utils.semantic_rank import sampled_gradient_diagnostics
            diagnostics = {key.replace("rank", "nfd"): value for key, value in
                           sampled_gradient_diagnostics(clean, weighted, net.parameters()).items()}
        combined = combined + weighted
        criterion.components.update(nfd.components)
    else:
        criterion.components["loss_nfd"] = 0.
    criterion.components["nfd_weight"] = strength
    clean_components = dict(criterion.components)
    distill_components = dict(nfd.components)
    if not torch.isfinite(combined):
        raise FloatingPointError("Non-finite clean SAM objective")
    combined.backward()
    clean_value = float(combined.detach())
    del clean, combined, output, features
    if strength > 0:
        del weighted
    after_rng = rng_state(device)
    parameters = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    if not parameters or any(not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Missing or non-finite clean SAM gradient")
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]))
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite SAM gradient norm")
    # Restore by copying snapshots: add/subtract would introduce rounding drift.
    originals = [p.detach().clone() for p in parameters]
    perturb_norm = 0.
    success = False
    try:
        with torch.no_grad():
            scale = rho / (norm + 1e-12)
            for p in parameters:
                p.add_(p.grad * scale)
            perturb_norm = float(torch.linalg.vector_norm(torch.stack([
                (p - original).norm() for p, original in zip(parameters, originals)])))
        optimizer.zero_grad(set_to_none=True)
        with second_pass_context(net, before_rng, after_rng, device):
            perturbed, perturbed_features = net.forward_with_features(images)
            perturbed_loss = criterion(perturbed, labels, indices, args)
            if strength > 0:
                perturbed_loss = perturbed_loss + strength * nfd.feature_loss(
                    perturbed_features, teacher_features, labels.argmax(1))
            if not torch.isfinite(perturbed_loss):
                raise FloatingPointError("Non-finite perturbed SAM objective")
            perturbed_loss.backward()
            perturbed_value = float(perturbed_loss.detach())
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in net.parameters()):
                raise FloatingPointError("Non-finite perturbed SAM gradient")
        success = True
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)
        criterion.components = clean_components
        nfd.components = distill_components
        if not success:
            optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    criterion.components.update(sam_clean_objective=clean_value,
                                sam_perturbed_objective=perturbed_value,
                                sam_objective_gap=perturbed_value - clean_value,
                                sam_gradient_norm=float(norm),
                                sam_perturbation_norm=perturb_norm,
                                sam_student_passes=2., sam_optimizer_updates=1.)
    return clean_value, diagnostics
