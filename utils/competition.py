"""Geometry-limited competition against fixed semantic hash centers."""
import math

import torch

from utils.reliable_relation import validate_similarity


class CenterCompetition(torch.nn.Module):
    def __init__(self, centers, similarity, temperature=0.1, m0=0.1, m1=0.4):
        super().__init__()
        if not all(math.isfinite(x) for x in (temperature, m0, m1)):
            raise ValueError("Competition parameters must be finite")
        if temperature <= 0 or min(m0, m1) < 0 or m0 + m1 > 1:
            raise ValueError("Require temperature > 0 and 0 <= m0 + m1 <= 1")
        if centers.ndim != 2 or min(centers.shape) < 1 or centers.shape[0] < 2:
            raise ValueError("Expected [classes>=2, bits>=1] centers")
        if not ((centers == -1) | (centers == 1)).all():
            raise ValueError("Centers must be binary {-1,+1}")
        validate_similarity(similarity, centers.shape[0])
        centers = centers.detach().clone()
        relation = similarity.detach().to(centers).clamp_min(0)
        distance = (centers.shape[1] - centers @ centers.T) / 2
        self.register_buffer("centers", centers)
        self.register_buffer("distance", distance)
        self.register_buffer("log_weight", (1 + relation).log())
        self.register_buffer("margin", distance / centers.shape[1] * (m0 + m1 * relation))
        self.temperature = temperature

    def forward(self, continuous, targets):
        scores = continuous @ self.centers.T / self.centers.shape[1]
        positive = scores.gather(1, targets[:, None])
        logits = self.log_weight[targets] + (
            scores - positive + self.margin[targets]
        ) / self.temperature
        logits = logits.scatter(1, targets[:, None], float("-inf"))
        return torch.logsumexp(
            torch.cat((torch.zeros_like(positive), logits), dim=1), dim=1
        ).mean()


def competition_weight(epoch, maximum, warmup=20, ramp_end=40):
    """One-based epoch: 1..20 off, 21..40 ramp, then constant."""
    if not math.isfinite(maximum) or maximum < 0 or warmup < 0 or ramp_end <= warmup:
        raise ValueError("Invalid competition schedule")
    return maximum * min(1.0, max(0.0, (epoch - warmup) / (ramp_end - warmup)))
