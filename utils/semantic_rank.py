"""I2-SBR: semantic weighting of binary ranking errors over a short train queue."""
import math

import torch
import torch.nn.functional as F

from utils.reliable_relation import tensor_sha256, validate_similarity

VERSION = "i2sbr-semantic-binary-ranking-v1"


def rank_weight(epoch, maximum=.05, warmup=20, ramp_end=40):
    if not math.isfinite(maximum) or maximum < 0 or not 0 <= warmup < ramp_end:
        raise ValueError("Invalid ranking weight or warmup/ramp schedule")
    return maximum * min(1., max(0., (epoch - warmup) / (ramp_end - warmup)))


def binary_code(values):
    return torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))


def proxy_distances(current, historical):
    """Exact binary forward, biased continuous gradient only to current codes."""
    historical = historical.detach().to(current)
    bits = current.shape[1]
    soft = (bits - current @ historical.T) / (2 * bits)
    hard = (bits - binary_code(current.detach()) @ binary_code(historical).T) / (2 * bits)
    # Parentheses give an exactly zero forward correction, including hinge boundaries.
    return hard + (soft - soft.detach()), hard, soft


class SemanticRank:
    def __init__(self, centers, similarity, capacity=1024, max_age=16,
                 mode="semantic", chunk_size=64):
        if centers.ndim != 2 or min(centers.shape) < 1 or len(centers) < 2:
            raise ValueError("Expected [classes>=2, bits>=1] centers")
        if not ((centers == -1) | (centers == 1)).all():
            raise ValueError("Centers must be binary")
        if len(torch.unique(centers, dim=0)) != len(centers):
            raise ValueError("Duplicate class centers")
        validate_similarity(similarity, len(centers))
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
               for x in (capacity, max_age, chunk_size)):
            raise ValueError("Queue capacity, age and chunk size must be positive integers")
        if mode not in ("semantic", "uniform"):
            raise ValueError("Unknown ranking mode")
        self.similarity = similarity.detach().cpu().double().clone()
        self.classes, self.bits = centers.shape
        self.capacity, self.max_age = capacity, max_age
        self.mode, self.chunk_size = mode, chunk_size
        self.identity = dict(version=VERSION, capacity=capacity, max_age=max_age,
                             mode=mode, chunk_size=chunk_size,
                             centers=tensor_sha256(centers), similarity=tensor_sha256(similarity))
        self.codes = torch.empty((0, self.bits))
        self.labels = torch.empty(0, dtype=torch.long)
        self.ids = torch.empty(0, dtype=torch.long)
        self.steps = torch.empty(0, dtype=torch.long)
        self.last_step = -1
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.stats = {name: 0 for name in (
            "batches", "anchors", "valid_anchors", "active_anchors", "positive_objects",
            "negative_objects", "triplets", "strict_errors", "ties",
            "continuous_correct_binary_wrong", "queue_items", "age_sum", "age_count",
            "near_negative_objects")}
        self.stats["max_observed_age"] = 0

    def _validate(self, codes, labels, ids):
        if codes.ndim != 2 or codes.shape[1] != self.bits or not codes.is_floating_point():
            raise ValueError("Expected continuous [batch,bits] codes")
        if not torch.isfinite(codes).all() or (codes.abs() > 1).any():
            raise ValueError("Continuous codes must be finite and in [-1,1]")
        for value in (labels, ids):
            if value.ndim != 1 or len(value) != len(codes) or value.dtype != torch.long:
                raise ValueError("Labels and stable sample IDs must be integer vectors")
        if ((labels < 0) | (labels >= self.classes)).any() or (ids < 0).any():
            raise ValueError("Invalid labels or sample IDs")

    def _keep(self, mask):
        self.codes, self.labels, self.ids, self.steps = (
            value[mask] for value in (self.codes, self.labels, self.ids, self.steps))

    def expire(self, step):
        if not isinstance(step, int) or step < self.last_step or step < 0:
            raise ValueError("Queue steps must be nonnegative and monotonic")
        self._keep(step - self.steps <= self.max_age)
        self.last_step = step

    @torch.no_grad()
    def enqueue(self, codes, labels, ids, step):
        self._validate(codes, labels, ids)
        self.expire(step)
        codes, labels, ids = (x.detach().cpu().clone() for x in (codes, labels, ids))
        # Last occurrence wins within a batch, as well as across batches.
        latest = {sample_id: i for i, sample_id in enumerate(ids.tolist())}
        selection = torch.tensor(sorted(latest.values()), dtype=torch.long)
        self._keep(torch.tensor([x not in latest for x in self.ids.tolist()], dtype=torch.bool))
        self.codes = torch.cat((self.codes.to(codes), codes[selection]))[-self.capacity:]
        self.labels = torch.cat((self.labels, labels[selection]))[-self.capacity:]
        self.ids = torch.cat((self.ids, ids[selection]))[-self.capacity:]
        self.steps = torch.cat((self.steps, torch.full((len(selection),), step, dtype=torch.long)))[-self.capacity:]

    def negative_weights(self, anchor_class, negative_labels):
        labels = negative_labels.detach().cpu()
        classes = sorted(set(labels.tolist()))
        if not classes or anchor_class in classes:
            raise ValueError("Expected nonempty unlike classes")
        near = sorted(classes, key=lambda c: (-self.similarity[anchor_class, c].item(), c))[:5]
        counts = {c: int((labels == c).sum()) for c in classes}
        mixture = .5 if self.mode == "semantic" else 0.
        weights = [( (1-mixture)/len(classes) + mixture * (c in near)/len(near)) / counts[c]
                   for c in labels.tolist()]
        return torch.tensor(weights, dtype=torch.float64), near

    def loss(self, current, labels, ids, step):
        self._validate(current, labels, ids)
        self.expire(step)
        self.stats["batches"] += 1
        self.stats["anchors"] += len(current)
        self.stats["queue_items"] += len(self.codes)
        ages = step - self.steps
        if len(ages):
            self.stats["age_sum"] += int(ages.sum())
            self.stats["age_count"] += len(ages)
            self.stats["max_observed_age"] = max(self.stats["max_observed_age"], int(ages.max()))
        if not len(self.codes):
            return current.sum() * 0
        distances, hard, soft = proxy_distances(current, self.codes)
        terms = []
        for i, (label, sample_id) in enumerate(zip(labels.detach().cpu().tolist(), ids.detach().cpu().tolist())):
            positive = torch.where((self.labels == label) & (self.ids != sample_id))[0]
            negative = torch.where((self.labels != label) & (self.ids != sample_id))[0]
            if not len(positive) or not len(negative):
                continue
            weights, near = self.negative_weights(label, self.labels[negative])
            weights = weights.to(current)
            p, n = positive.to(current.device), negative.to(current.device)
            value = current[i].sum() * 0
            active = False
            for start in range(0, len(p), self.chunk_size):
                pc = p[start:start+self.chunk_size]
                # Normalize over ALL positives and negatives, including satisfied zeros.
                value = value + (F.relu(1/self.bits + distances[i, pc, None] - distances[i, n][None, :])
                                 * weights[None, :]).sum() / len(p)
                with torch.no_grad():
                    gap = hard[i, pc, None] - hard[i, n][None, :]
                    errors, ties = gap > 0, gap == 0
                    active = active or bool((gap >= 0).any())
                    self.stats["strict_errors"] += int(errors.sum())
                    self.stats["ties"] += int(ties.sum())
                    self.stats["triplets"] += gap.numel()
                    self.stats["continuous_correct_binary_wrong"] += int(
                        ((soft[i, pc, None] < soft[i, n][None, :]) & errors).sum())
            terms.append(value)
            self.stats["valid_anchors"] += 1
            self.stats["active_anchors"] += int(active)
            self.stats["positive_objects"] += len(p)
            self.stats["negative_objects"] += len(n)
            self.stats["near_negative_objects"] += sum(c in near for c in self.labels[negative].tolist())
        return torch.stack(terms).mean() if terms else current.sum() * 0

    def diagnostics(self):
        result = dict(self.stats)
        def ratio(a, b):
            return self.stats[a] / self.stats[b] if self.stats[b] else None
        result.update(valid_anchor_fraction=ratio("valid_anchors", "anchors"),
                      strict_error_rate=ratio("strict_errors", "triplets"),
                      tie_rate=ratio("ties", "triplets"),
                      mean_age=ratio("age_sum", "age_count"),
                      queue_size=len(self.ids), version=VERSION)
        return result

    def state_dict(self):
        return dict(identity=dict(self.identity), codes=self.codes.clone(),
                    labels=self.labels.clone(), ids=self.ids.clone(), steps=self.steps.clone(),
                    last_step=self.last_step)

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError("Ranking queue configuration/content mismatch")
        self._validate(state["codes"], state["labels"], state["ids"])
        steps = state["steps"]
        if (not isinstance(state["last_step"], int) or state["last_step"] < -1
                or steps.dtype != torch.long or steps.shape != state["ids"].shape
                or len(steps) > self.capacity or len(torch.unique(state["ids"])) != len(steps)
                or (steps < 0).any() or (steps > state["last_step"]).any()
                or (steps[1:] < steps[:-1]).any()
                or (state["last_step"] - steps > self.max_age).any()):
            raise ValueError("Invalid saved ranking queue")
        for key in ("codes", "labels", "ids", "steps"):
            setattr(self, key, state[key].detach().cpu().clone())
        self.last_step = int(state["last_step"])


def sampled_gradient_diagnostics(clean, weighted_rank, parameters):
    """Parameter-space norms for one batch, without changing .grad or RNG."""
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
