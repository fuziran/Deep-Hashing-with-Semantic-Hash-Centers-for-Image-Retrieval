"""I3: frozen teacher covariance, protected directions, analytic bit BCE increment.

Only the training relation loader is allowed during preparation. All modes have
unit covariance trace, so rho is the RMS perturbation / student feature norm.
No random feature samples, additional student forwards, or retrieval changes.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.reliable_relation import tensor_sha256

VERSION = "i3-npa-v1"
MODES = ("protected", "covariance", "isotropic")


def npa_weight(epoch, maximum=1., warmup=40, ramp_end=60):
    if (not isinstance(epoch, int) or epoch < 1 or not math.isfinite(maximum)
            or not 0 <= maximum <= 1 or not isinstance(warmup, int)
            or not isinstance(ramp_end, int) or not 0 <= warmup < ramp_end):
        raise ValueError("Require NPA weight in [0,1] and 0 <= warmup < ramp-end")
    return maximum * min(1., max(0., (epoch - warmup) / (ramp_end - warmup)))


class ClassCovariance:
    """Float64 CPU parallel-Welford moments; one observation per training image."""
    def __init__(self, classes, dim):
        self.counts = torch.zeros(classes, dtype=torch.long)
        self.means = torch.zeros(classes, dim, dtype=torch.float64)
        self.m2 = torch.zeros(classes, dim, dim, dtype=torch.float64)

    def update(self, features, targets):
        x, y = features.detach().cpu().double(), targets.detach().cpu()
        if (x.ndim != 2 or x.shape[1] != self.means.shape[1]
                or not torch.isfinite(x).all() or len(x) == 0
                or y.shape != (len(x),) or y.dtype != torch.long
                or (y < 0).any() or (y >= len(self.counts)).any()):
            raise ValueError("Invalid covariance features/targets")
        for label in y.unique().tolist():
            batch = x[y == label]
            n, old = len(batch), int(self.counts[label])
            mean = batch.mean(0)
            centered = batch - mean
            delta = mean - self.means[label]
            self.m2[label] += centered.T @ centered + torch.outer(delta, delta) * (old*n/(old+n))
            self.means[label] += delta * (n/(old+n))
            self.counts[label] += n

    def finish(self, shrinkage=.2):
        if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
            raise ValueError("NPA shrinkage must be finite in [0,1]")
        if (self.counts < 2).any():
            raise ValueError("NPA needs at least two training images in every class")
        pooled = self.m2.sum(0) / (self.counts - 1).sum()
        covariance = self.m2 / (self.counts - 1)[:, None, None]
        return (1-shrinkage)*covariance + shrinkage*pooled[None]


def prepare_covariances(covariance, basis, mode="protected"):
    if mode not in MODES:
        raise ValueError("Unknown NPA mode")
    c = covariance.detach().cpu().double()
    q = basis.detach().cpu().double()
    if (c.ndim != 3 or c.shape[1] != c.shape[2] or q.ndim != 3
            or q.shape[:2] != c.shape[:2] or not torch.isfinite(c).all()
            or not torch.isfinite(q).all()):
        raise ValueError("Invalid NPA covariance/basis shape or values")
    classes, dim, _ = c.shape
    result, retained = [], []
    eye = torch.eye(dim, dtype=torch.float64)
    for label in range(classes):
        matrix = (c[label] + c[label].T) / 2
        trace = matrix.trace()
        if trace <= 1e-12:
            raise ValueError("NPA class covariance has no usable variation")
        # Preparation is once per run, not inside training. Reject corrupted PSD
        # statistics rather than clamping negative quadratic forms silently.
        if torch.linalg.eigvalsh(matrix).min() < -1e-9 * max(1., trace.item()):
            raise ValueError("NPA covariance must be positive semidefinite")
        if mode == "protected":
            active = q[label][:, q[label].square().sum(0) > 1e-12]
            if active.shape[1] and not torch.allclose(
                    active.T @ active, torch.eye(active.shape[1], dtype=torch.float64),
                    atol=1e-5, rtol=1e-5):
                raise ValueError("NPA protection basis must be orthonormal")
            complement = eye - active @ active.T
            matrix = complement @ matrix @ complement
        ratio = matrix.trace() / trace
        retained.append(float(ratio))
        if ratio <= 1e-6:
            raise ValueError("NPA protection removed all usable variation")
        matrix = eye/dim if mode == "isotropic" else matrix/matrix.trace()
        result.append(((matrix+matrix.T)/2).float())
    return torch.stack(result), retained


class NeighborhoodProtectedAugmentation(nn.Module):
    def __init__(self, covariance, rho=.05):
        super().__init__()
        if not math.isfinite(rho) or rho < 0:
            raise ValueError("NPA rho must be finite and nonnegative")
        if (covariance.ndim != 3 or covariance.shape[1] != covariance.shape[2]
                or not covariance.is_floating_point() or not torch.isfinite(covariance).all()
                or not torch.allclose(covariance, covariance.transpose(1,2), atol=1e-6)
                or not torch.allclose(covariance.diagonal(dim1=1,dim2=2).sum(1),
                                      covariance.new_ones(covariance.shape[0]), atol=1e-5)):
            raise ValueError("NPA requires symmetric unit-trace covariances")
        self.register_buffer("covariance", covariance.detach().clone())
        self.rho = rho
        self.components = {}

    def forward(self, output, features, targets, hash_weight, centers):
        classes, dim, _ = self.covariance.shape
        if (output.ndim != 2 or features.shape != (len(output),dim)
                or hash_weight.shape != (output.shape[1],dim)
                or centers.shape != (classes,output.shape[1]) or len(output) == 0
                or targets.shape != (len(output),) or targets.dtype != torch.long
                or (targets < 0).any() or (targets >= classes).any()
                or not ((centers == -1) | (centers == 1)).all()
                or not all(torch.isfinite(t).all() for t in (output,features,hash_weight))):
            raise ValueError("Invalid NPA outputs/features/head/targets/centers")
        if self.rho == 0:
            return output.sum()*0
        # Compute each class once; do not materialize a [batch,dim,dim] tensor.
        present, inverse = targets.unique(sorted=True, return_inverse=True)
        class_variance = torch.stack([
            ((hash_weight @ self.covariance[y].to(hash_weight)) * hash_weight).sum(1)
            for y in present.tolist()
        ])
        variance = class_variance[inverse].clamp_min(0)
        variance = variance * features.detach().square().sum(1,keepdim=True) * self.rho**2
        signed = -2 * centers[targets].to(output) * output
        # Float64 subtraction retains very small increments on saturated logits.
        increment = (F.softplus(signed.double()+2*variance.double())
                     - F.softplus(signed.double())).clamp_min(0).mean().to(output)
        if not torch.isfinite(increment):
            raise FloatingPointError("Non-finite NPA increment")
        self.components = dict(loss_npa=increment.detach().item(),
                               npa_logit_variance=variance.detach().mean().item(),
                               npa_positive_variance_fraction=(variance.detach()>0).float().mean().item())
        return increment


@torch.no_grad()
def prepare_npa(args, nfd, relation_loader, train_loader):
    """Build only from the exact training subset, preserving global RNG states."""
    from utils.experiment import stable_config_hash
    if args.npa_weight == 0 or args.npa_rho == 0:
        args.npa_disabled = True
        return None
    if (getattr(relation_loader, "persistent_workers", False)
            or getattr(train_loader, "persistent_workers", False)):
        raise ValueError("NPA preparation requires non-persistent workers")
    # Require the audited CIFAR loader's exact image IDs, not just equal lengths.
    import numpy as np
    train_ids = getattr(train_loader.dataset, "indices", None)
    relation_ids = getattr(relation_loader.dataset, "indices", None)
    if train_ids is None or relation_ids is None or not np.array_equal(train_ids, relation_ids):
        raise ValueError("NPA statistics must use exactly the training image IDs")
    moments = ClassCovariance(args.num_classes, nfd.dim)
    observed = set()
    print(f"NPA: estimating two-view teacher covariance from {len(train_ids)} training images only", flush=True)
    import random
    python_state, numpy_state = random.getstate(), np.random.get_state()
    generator = getattr(relation_loader, "generator", None)
    generator_state = generator.get_state() if generator is not None else None
    try:
        # Deterministic transforms and eval teacher draw no CUDA randomness.
        # DataLoader worker base seeds consume CPU RNG, restored by this context.
        with torch.random.fork_rng(devices=[]):
            nfd.teacher.eval()
            for batch_index, (images, labels, ids) in enumerate(relation_loader):
                ids = ids.tolist()
                if len(set(ids)) != len(ids) or observed.intersection(ids):
                    raise ValueError("Duplicate training IDs in NPA statistics")
                observed.update(ids)
                if (labels.ndim != 2 or labels.shape[1] != args.num_classes
                        or not ((labels == 0) | (labels == 1)).all()
                        or not (labels.sum(1) == 1).all()):
                    raise ValueError("NPA currently requires one-hot single-label targets")
                images = images.to(args.device, non_blocking=True)
                first = nfd.teacher.forward_features(images)
                second = nfd.teacher.forward_features(images.flip(-1))
                if (not torch.isfinite(first).all() or not torch.isfinite(second).all()
                        or (first.norm(dim=1) <= 1e-12).any() or (second.norm(dim=1) <= 1e-12).any()):
                    raise ValueError("Invalid teacher features during NPA preparation")
                features = (F.normalize(first,dim=1) + F.normalize(second,dim=1))/2
                if (features.norm(dim=1) <= 1e-12).any():
                    raise ValueError("NPA two-view teacher feature has zero norm")
                moments.update(F.normalize(features,dim=1), labels.argmax(1))
                if (batch_index+1) % 25 == 0:
                    print(f"NPA statistics: {len(observed)}/{len(train_ids)} images", flush=True)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if generator is not None:
            generator.set_state(generator_state)
    if observed != set(map(int,train_ids)):
        raise ValueError("NPA statistics did not cover exactly the training subset")
    covariance, retained = prepare_covariances(moments.finish(args.npa_shrinkage),
                                               nfd.basis, args.npa_mode)
    identity = dict(version=VERSION, mode=args.npa_mode, rho=args.npa_rho,
                    shrinkage=args.npa_shrinkage, teacher_sha256=args.nfd_teacher_sha256,
                    basis_sha256=args.nfd_basis_sha256, split_hash=args.split_hash,
                    counts=moments.counts.tolist(), retained_trace_fraction=retained,
                    covariance_sha256=tensor_sha256(covariance),
                    policy="unit-trace-two-view-normalized-teacher-detached-student-norm")
    identity["statistics_id"] = stable_config_hash(identity)
    args.npa_covariance_sha256 = identity["covariance_sha256"]
    args.npa_statistics_id = identity["statistics_id"]
    args.npa_class_counts = identity["counts"]
    args.npa_retained_trace_fraction = retained
    module = NeighborhoodProtectedAugmentation(covariance, args.npa_rho).to(args.device)
    module.artifact = dict(identity=identity, covariance=covariance,
                           teacher_means=moments.means.float())
    print(f"NPA: prepared {args.npa_mode} statistics {args.npa_statistics_id}; "
          f"retained trace range={min(retained):.4f}..{max(retained):.4f}", flush=True)
    return module
