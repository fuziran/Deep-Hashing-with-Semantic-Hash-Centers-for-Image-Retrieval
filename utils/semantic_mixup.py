"""I2-M: frozen-center input Mixup; all pairing randomness is local to CPU."""
from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F

from utils.reliable_relation import validate_similarity

VERSION = "i2m-semantic-center-mixup-v1"


def mix_weight(epoch, maximum=.1, warmup=20, ramp_end=40):
    if not math.isfinite(maximum) or maximum < 0 or not 0 <= warmup < ramp_end:
        raise ValueError("Invalid Mixup weight or warmup/ramp schedule")
    return maximum * min(1., max(0., (epoch - warmup) / (ramp_end - warmup)))


@contextmanager
def frozen_batchnorm(net):
    """Use clean running statistics, retaining affine gradients and other modes."""
    layers = [(m, m.training) for m in net.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    try:
        for layer, _ in layers:
            layer.train(False)
        yield
    finally:
        for layer, training in layers:
            layer.train(training)


def soft_center_loss(logits, targets):
    """Soft BCE only: the mixed branch must not receive hard quantization."""
    if logits.shape != targets.shape or logits.ndim != 2 or not logits.numel():
        raise ValueError("Expected nonempty matching [batch, bits] logits/targets")
    if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError("Soft targets must be finite probabilities")
    loss = F.binary_cross_entropy_with_logits(2 * logits, targets)
    # Clamp only the logarithm, preserving exact zero entropy at hard endpoints.
    tiny = torch.finfo(targets.dtype).tiny
    entropy = -(targets * targets.clamp_min(tiny).log()
                + (1 - targets) * (1 - targets).clamp_min(tiny).log()).mean()
    return loss, entropy.detach()


class SemanticMixup:
    def __init__(self, centers, similarity, seed, mode="semantic", budget=None):
        if centers.ndim != 2 or centers.shape[0] < 2 or centers.shape[1] < 1:
            raise ValueError("Expected [classes>=2, bits>=1] centers")
        if not ((centers == -1) | (centers == 1)).all():
            raise ValueError("Centers must contain only -1/+1")
        validate_similarity(similarity, len(centers))
        if mode not in ("semantic", "plain", "uniform_budgeted"):
            raise ValueError("Unknown Mixup mode")
        self.centers = centers.detach().cpu().float().clone()
        self.similarity = similarity.detach().cpu().double().clone()
        self.distance = (self.centers[:, None] != self.centers[None, :]).sum(-1)
        if (self.distance[~torch.eye(len(centers), dtype=torch.bool)] == 0).any():
            raise ValueError("Mixup cannot distinguish duplicate class centers")
        self.budget = centers.shape[1] / 16 if budget is None else budget
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("Mixup distance budget must be finite and positive")
        self.mode = mode
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.reset_diagnostics()

    def state_dict(self):
        return {"version": VERSION, "mode": self.mode, "budget": self.budget,
                "rng_state": self.generator.get_state().clone()}

    def load_state_dict(self, state):
        if (state["version"], state["mode"], state["budget"]) != (VERSION, self.mode, self.budget):
            raise ValueError("Mixup state configuration mismatch")
        self.generator.set_state(state["rng_state"].cpu())

    def reset_diagnostics(self):
        self.samples = []
        self.pair_counts = torch.zeros_like(self.distance)
        self.attempted = 0

    def class_probabilities(self, anchor, available):
        """Class-uniform base; ties in semantic top-five use ascending class ID."""
        classes = sorted(c for c in available if c != anchor)
        if not classes:
            return classes, torch.empty(0, dtype=torch.float64)
        probabilities = torch.full((len(classes),), 1 / len(classes), dtype=torch.float64)
        if self.mode == "semantic":
            nearest = sorted(classes, key=lambda c: (-self.similarity[anchor, c].item(), c))[:5]
            probabilities *= .5
            for c in nearest:
                probabilities[classes.index(c)] += .5 / len(nearest)
        return classes, probabilities

    @torch.no_grad()
    def sample(self, labels):
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError("Mixup requires integer class indices")
        y = labels.detach().cpu()
        if ((y < 0) | (y >= len(self.centers))).any():
            raise ValueError("Invalid Mixup class index")
        available = sorted(set(y.tolist()))
        members = {c: torch.where(y == c)[0] for c in available}
        choices = {c: self.class_probabilities(c, available) for c in available}
        anchors, partners, fractions, rows = [], [], [], []
        self.attempted += len(y)
        for i, anchor in enumerate(y.tolist()):
            classes, probabilities = choices[anchor]
            if not classes:
                continue
            if self.mode == "plain":
                # Ordinary control samples uniformly among all unlike examples.
                probabilities = torch.tensor([len(members[c]) for c in classes], dtype=torch.float64)
                probabilities /= probabilities.sum()
            c = classes[torch.multinomial(probabilities, 1, generator=self.generator).item()]
            j = members[c][torch.randint(len(members[c]), (1,), generator=self.generator).item()].item()
            beta0 = torch.rand((), generator=self.generator).item() * .5
            distance = self.distance[anchor, c].item()
            beta = min(beta0, self.budget / distance) if self.mode != "plain" else beta0
            anchors.append(i)
            partners.append(j)
            fractions.append(beta)
            rows.append((beta, distance, self.similarity[anchor, c].item(), float(beta < beta0)))
            self.pair_counts[anchor, c] += 1
        if rows:
            self.samples.append(torch.tensor(rows, dtype=torch.float64))
        return (torch.tensor(anchors, dtype=torch.long), torch.tensor(partners, dtype=torch.long),
                torch.tensor(fractions, dtype=torch.float32))

    def make_batch(self, images, labels):
        anchors, partners, beta = self.sample(labels)
        a, b = anchors.to(images.device), partners.to(images.device)
        fraction = beta.to(images).reshape((-1,) + (1,) * (images.ndim - 1))
        mixed = (1 - fraction) * images[a] + fraction * images[b]
        y = labels.detach().cpu()
        targets = (self.centers + 1) / 2
        soft = ((1 - beta[:, None]) * targets[y[anchors]]
                + beta[:, None] * targets[y[partners]])
        return mixed, soft.to(images)

    def diagnostics(self):
        result = {"attempted": self.attempted, "valid_pairs": int(self.pair_counts.sum()),
                  "pair_counts": self.pair_counts.tolist()}
        if self.samples:
            rows = torch.cat(self.samples)
            for col, name in enumerate(("beta", "center_distance", "similarity", "cap_triggered")):
                values = rows[:, col]
                result[name] = {"mean": values.mean().item(), "quantiles": torch.quantile(
                    values, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)).tolist()}
        return result


def backward_mix(net, images, labels, sampler, weight, sample_gradients=False):
    """Accumulate auxiliary gradients after clean backward, before the sole step."""
    result = {"loss_mix": 0., "mix_target_entropy": 0., "mix_kl": 0.,
              "mix_weight": weight}
    if weight == 0:
        return result
    mixed, targets = sampler.make_batch(images, labels)
    if not len(mixed):
        return result
    parameters = [p for p in net.parameters() if p.requires_grad]
    before = [None if p.grad is None else p.grad.detach().clone() for p in parameters] \
        if sample_gradients else None
    with frozen_batchnorm(net):
        loss, entropy = soft_center_loss(net(mixed), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite mixed soft-center loss")
        (weight * loss).backward()
    result.update(loss_mix=loss.detach().item(), mix_target_entropy=entropy.item(),
                  mix_kl=(loss.detach() - entropy).item())
    if before is not None:
        clean_squared, mixed_squared = 0., 0.
        for p, old in zip(parameters, before):
            if old is not None:
                clean_squared += old.double().square().sum().item()
            if p.grad is not None:
                delta = p.grad.detach() if old is None else p.grad.detach() - old
                mixed_squared += delta.double().square().sum().item()
        result["sampled_clean_grad_norm"] = math.sqrt(clean_squared)
        result["sampled_weighted_mix_grad_norm"] = math.sqrt(mixed_squared)
    return result
