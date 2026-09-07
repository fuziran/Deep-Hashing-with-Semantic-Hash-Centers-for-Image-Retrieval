"""I2-R: frozen class/bit risk weights, estimated from training forwards only."""
import math

import torch

from utils.reliable_relation import state_sha256, validate_similarity

VERSION = "i2r-bit-supervision-v1"


def risk_schedule(epoch, maximum=.25, stats_start=16, warmup=20, ramp_end=40):
    if (not math.isfinite(maximum) or maximum < 0
            or not 1 <= stats_start <= warmup < ramp_end):
        raise ValueError("Invalid I2-R rho or statistics/warmup/ramp schedule")
    return maximum * min(1., max(0., (epoch - warmup) / (ramp_end - warmup)))


class BitSupervision(torch.nn.Module):
    def __init__(self, centers, similarity, stats_start=16, warmup=20):
        super().__init__()
        risk_schedule(1, stats_start=stats_start, warmup=warmup, ramp_end=warmup + 1)
        if centers.ndim != 2 or centers.shape[0] < 2 or centers.shape[1] < 1:
            raise ValueError("Expected [classes>=2, bits>=1] centers")
        if not ((centers == -1) | (centers == 1)).all():
            raise ValueError("Centers must be binary {-1,+1}")
        validate_similarity(similarity, centers.shape[0])
        # CPU float64 aggregation avoids CUDA scatter nondeterminism and RNG use.
        h = centers.detach().to(device="cpu", dtype=torch.float64).clone()
        different = h[:, None, :] != h[None, :, :]
        distance = different.sum(-1)
        diagonal = torch.eye(len(h), dtype=torch.bool)
        if (distance[~diagonal] == 0).any():
            raise ValueError("I2-R cannot distinguish duplicate class centers")
        relation = similarity.detach().to(device="cpu", dtype=torch.float64)
        pair_weight = (1 + relation.clamp_min(0)).masked_fill(diagonal, 0)
        pair_weight = pair_weight / pair_weight.sum(1, keepdim=True)
        safe_distance = distance.masked_fill(diagonal, 1)
        geometry = (pair_weight[:, :, None] * different / safe_distance[:, :, None]).sum(1)
        self.register_buffer("centers", h)
        self.register_buffer("geometry", geometry)
        self.register_buffer("deviation_sum", torch.zeros_like(h))
        self.register_buffer("count", torch.zeros(len(h), dtype=torch.long))
        self.register_buffer("deviation", torch.zeros_like(h))
        self.register_buffer("risk", torch.zeros_like(h))
        self.register_buffer("frozen", torch.tensor(False))
        self.stats_start, self.warmup = stats_start, warmup

    @torch.no_grad()
    def observe(self, continuous, targets, epoch):
        if not self.stats_start <= epoch <= self.warmup:
            return
        if self.frozen.item():
            raise RuntimeError("I2-R statistics are already frozen")
        u = continuous.detach().to(device="cpu", dtype=torch.float64)
        y = targets.detach().to(device="cpu", dtype=torch.long)
        if u.shape != (len(y), self.centers.shape[1]):
            raise ValueError("Invalid I2-R observation shape")
        if not torch.isfinite(u).all() or (u.abs() > 1).any():
            raise ValueError("I2-R observations must be finite and in [-1,1]")
        if ((y < 0) | (y >= len(self.centers))).any():
            raise ValueError("Invalid I2-R class index")
        errors = (1 - self.centers[y] * u) / 2
        self.deviation_sum.index_add_(0, y, errors)
        self.count.index_add_(0, y, torch.ones_like(y))

    @torch.no_grad()
    def freeze(self):
        if self.frozen.item():
            return
        if (self.count == 0).any():
            missing = torch.where(self.count == 0)[0].tolist()
            raise ValueError(f"Missing I2-R training statistics for classes: {missing}")
        self.deviation.copy_(self.deviation_sum / self.count[:, None])
        product = self.geometry * self.deviation
        self.risk.copy_(product / (product.max(1, keepdim=True).values + 1e-12))
        self.frozen.fill_(True)

    def weights(self, targets, rho, reference):
        if not math.isfinite(rho) or rho < 0:
            raise ValueError("I2-R rho must be finite and nonnegative")
        if rho == 0:
            return reference.new_ones((len(targets), self.centers.shape[1]))
        if not self.frozen.item():
            raise RuntimeError("Freeze I2-R statistics before enabling weights")
        weights = 1 + rho * self.risk
        weights = weights / weights.mean(1, keepdim=True)
        return weights.to(reference)[targets]

    def diagnostics(self, maximum):
        if not self.frozen.item():
            raise RuntimeError("Cannot save unfrozen I2-R statistics")
        tensors = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
        weights = 1 + maximum * self.risk
        tensors["weights_at_maximum"] = (weights / weights.mean(1, keepdim=True)).cpu()
        summary = {}
        for name in ("geometry", "deviation", "risk", "weights_at_maximum"):
            values = tensors[name].double().flatten()
            summary[name] = dict(zip(
                ("min", "p25", "median", "p75", "max"),
                torch.quantile(values, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)).tolist(),
            ))
            summary[name]["mean"] = values.mean().item()
            summary[name]["std"] = values.std(unbiased=False).item()
        return {"version": VERSION, "stats_start": self.stats_start,
                "stats_end": self.warmup, "rho_max": maximum,
                "content_sha256": state_sha256(tensors),
                "summary": summary, "tensors": tensors}
