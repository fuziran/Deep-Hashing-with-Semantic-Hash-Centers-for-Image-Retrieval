"""Frozen I1 teacher and neighborhood-weighted pre-hash feature distillation."""
import argparse
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from utils.reliable_relation import state_sha256, tensor_sha256, validate_similarity

VERSION = "i2nfd-neighborhood-feature-distillation-v1"
RELIABLE_VERSION = "i3-reliability-gated-neighborhood-distillation-v1"


def nfd_weight(epoch, maximum=.01, warmup=20, ramp_end=40):
    if (not isinstance(epoch, int) or epoch < 1 or not math.isfinite(maximum)
            or maximum < 0 or not isinstance(warmup, int) or not isinstance(ramp_end, int)
            or not 0 <= warmup < ramp_end):
        raise ValueError("Invalid NFD weight schedule")
    return maximum * min(1., max(0., (epoch - warmup) / (ramp_end - warmup)))


class NeighborhoodFeatureDistillation(nn.Module):
    def __init__(self, teacher, similarity, prototypes, alpha=.75, topk=5):
        super().__init__()
        if not math.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("NFD alpha must be finite in [0, 1)")
        if (not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2
                or not prototypes.is_floating_point() or min(prototypes.shape) < 2
                or not torch.isfinite(prototypes).all()):
            raise ValueError("Expected finite floating [classes, features] prototypes")
        classes, dim = prototypes.shape
        validate_similarity(similarity, classes)
        if not isinstance(topk, int) or not 0 < topk < classes:
            raise ValueError("NFD topk must be positive and smaller than class count")
        proto = prototypes.detach().cpu().double()
        if not torch.allclose(proto.norm(dim=1), torch.ones(classes, dtype=torch.double), atol=1e-5, rtol=1e-5):
            raise ValueError("I1 prototypes must be unit-normalized")
        sim = similarity.detach().cpu().double()
        basis = torch.zeros(classes, dim, min(topk, dim), dtype=torch.double)
        ranks, neighbors = [], []
        for y in range(classes):
            near = sorted((c for c in range(classes) if c != y),
                          key=lambda c: (-sim[y, c].item(), c))[:topk]
            u, singular, _ = torch.linalg.svd((proto[y] - proto[near]).T, full_matrices=False)
            threshold = max(1e-12, 1e-8 * singular.max().item())
            rank = int((singular > threshold).sum())
            basis[y, :, :rank] = u[:, :rank]
            ranks.append(rank)
            neighbors.append(near)
        self.teacher = teacher.requires_grad_(False).eval()
        for parameter in self.teacher.parameters():
            parameter.grad = None
        self.alpha, self.dim, self.classes = alpha, dim, classes
        self.register_buffer("basis", basis.float())
        self.register_buffer("ranks", torch.tensor(ranks, dtype=torch.long))
        self.register_buffer("neighbors", torch.tensor(neighbors, dtype=torch.long))
        self.identity = dict(version=VERSION, alpha=alpha, topk=topk,
                             prototypes_sha256=tensor_sha256(prototypes),
                             similarity_sha256=tensor_sha256(similarity),
                             basis_sha256=tensor_sha256(self.basis),
                             ranks=ranks, feature_dim=dim)
        self.components = {}
        self.gate_mode = None
        self.last_teacher_features = None

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def feature_loss(self, student, teacher, targets, gate=None):
        if (student.ndim != 2 or student.shape != teacher.shape or student.shape[1] != self.dim
                or len(student) == 0 or not student.is_floating_point() or not teacher.is_floating_point()
                or not torch.isfinite(student).all() or not torch.isfinite(teacher).all()):
            raise ValueError("Invalid NFD feature shape or values")
        if (targets.shape != (len(student),) or targets.dtype != torch.long
                or (targets < 0).any() or (targets >= self.classes).any()):
            raise ValueError("Invalid NFD class targets")
        if (student.norm(dim=1) <= 1e-12).any() or (teacher.norm(dim=1) <= 1e-12).any():
            raise ValueError("Zero-norm NFD features")
        teacher = teacher.detach().to(student)
        e = F.normalize(student, dim=1, eps=1e-12) - F.normalize(teacher, dim=1, eps=1e-12)
        targets = targets.to(self.ranks.device)
        q = self.basis[targets].to(student)
        ordinary = e.square().sum(1)
        projected = torch.bmm(e[:, None, :], q).squeeze(1).square().sum(1)
        ranks = self.ranks[targets].to(student)
        if gate is not None:
            if (gate.shape != (len(student),) or not gate.is_floating_point()
                    or not torch.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any()):
                raise ValueError("NFD reliability gate must be finite in [0,1] per sample")
            alpha = self.alpha * gate.detach().to(student)
            values = ((1-alpha)*ordinary + alpha*projected) / (
                (1-alpha) + alpha*ranks/self.dim)
            values = torch.where(ranks == 0, ordinary, values)
        elif self.alpha == 0:
            values = ordinary
        else:
            values = ((1-self.alpha)*ordinary + self.alpha*projected) / (
                (1-self.alpha) + self.alpha*ranks/self.dim)
            values = torch.where(ranks == 0, ordinary, values)
        loss = values.mean()
        self.components = dict(loss_nfd=loss.detach().item(),
                               nfd_feature_mse=ordinary.detach().mean().item(),
                               nfd_projected_error=projected.detach().mean().item(),
                               nfd_mean_rank=ranks.detach().mean().item())
        return loss

    def forward(self, student_features, images, targets):
        self.teacher.eval()
        with torch.no_grad():
            teacher_features = self.teacher.forward_features(images)
            self.last_teacher_features = teacher_features.detach()
            if self.gate_mode is not None:
                if self.gate_mode not in ("gated", "mean", "fixed"):
                    raise ValueError("Unknown reliability-gate control mode")
                original = self.teacher.classify_layer(teacher_features).argmax(1)
                flipped_features = self.teacher.forward_features(torch.flip(images, dims=(-1,)))
                flipped = self.teacher.classify_layer(flipped_features).argmax(1)
                raw_gate = (original == targets) & (flipped == targets)
        if self.gate_mode is None:
            return self.feature_loss(student_features, teacher_features, targets)
        if self.gate_mode == "fixed":
            loss = self.feature_loss(student_features, teacher_features, targets)
            effective = raw_gate.new_ones(raw_gate.shape, dtype=student_features.dtype)
        else:
            effective = raw_gate.to(student_features.dtype)
            if self.gate_mode == "mean":
                effective = effective.mean().expand_as(effective)
            loss = self.feature_loss(student_features, teacher_features, targets, effective)
        self.components.update(
            rgnfd_teacher_correct=float((original == targets).float().mean()),
            rgnfd_flipped_teacher_correct=float((flipped == targets).float().mean()),
            rgnfd_raw_gate_fraction=float(raw_gate.float().mean()),
            rgnfd_effective_gate_fraction=float(effective.float().mean()),
        )
        return loss


def load_nfd(args):
    """Load trusted parent artifacts only; never train or download a teacher."""
    import json
    import network
    from torchvision import models
    from utils.experiment import build_cache_metadata, load_cache
    parent = json.loads(Path(args.parent_b1_config).read_text(encoding="utf-8"))
    parent_args = argparse.Namespace(**parent)
    artifact = load_cache(args.similarity_cache, build_cache_metadata(parent_args, "similarity"))
    state = load_cache(args.teacher_cache, build_cache_metadata(parent_args, "classifier"))
    if artifact is None or state is None:
        raise FileNotFoundError("NFD requires the existing parent relation and classifier caches")
    if state_sha256(state) != parent["classifier_sha256"]:
        raise ValueError("NFD teacher content does not match parent B1")
    if artifact.get("classifier_sha256") != parent["classifier_sha256"]:
        raise ValueError("NFD prototype teacher provenance mismatch")
    if tensor_sha256(artifact["similarity"]) != parent["similarity_content_sha256"]:
        raise ValueError("NFD relation content mismatch")
    if artifact["prototypes"].shape != (args.num_classes, 512):
        raise ValueError("NFD requires I1 ResNet34 512-dimensional prototypes")
    # Construction is CPU-only, so preserve CPU RNG; subsequent .to does not sample.
    with torch.random.fork_rng(devices=[]), patch.object(
            network, "_build_resnet", side_effect=lambda *a, **kw: models.resnet34(weights=None)):
        teacher = network.ClassifyNet(args.num_classes)
    teacher.load_state_dict(state, strict=True)
    del state
    module = NeighborhoodFeatureDistillation(teacher, artifact["similarity"],
                                            artifact["prototypes"], args.nfd_alpha, args.nfd_topk)
    if getattr(args, "variant", None) == "b3rnd":
        module.gate_mode = args.rgnfd_mode
    args.nfd_teacher_sha256 = parent["classifier_sha256"]
    args.nfd_prototypes_sha256 = module.identity["prototypes_sha256"]
    args.nfd_basis_sha256 = module.identity["basis_sha256"]
    args.nfd_projection_ranks = module.identity["ranks"]
    args.nfd_relation_file_sha256 = hashlib.sha256(Path(args.similarity_cache).read_bytes()).hexdigest()
    args.nfd_teacher_config_hash = build_cache_metadata(parent_args, "classifier")["config_hash"]
    return module.to(args.device)
