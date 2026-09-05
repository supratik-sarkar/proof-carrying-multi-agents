"""Replay-interventional attribution (NOT causal root-cause identification).

Resp(e) is the total effect of masking component e under the replay
intervention on the logged graph. It is non-additive, does not sum to one, and
is not a blame share. When the observed top-two margin falls below

    2 * sqrt(2 * log(2|U|/delta) / M)

the ranking is reported as Unresolved rather than forced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class Attribution:
    ranking: List[str]
    scores: Dict[str, float]
    top1: Optional[str]
    top3: List[str]
    margin: Optional[float]
    unresolved: bool
    threshold: float
    M: int
    n_components: int
    semantics: str = ("replay-interventional attribution under the logged replay model; "
                      "certificate-sensitive component, not real-world causal root cause")

    def to_dict(self) -> dict:
        return vars(self)


def unresolved_threshold(n_components: int, M: int, delta: float = 0.05) -> float:
    if M <= 0 or n_components <= 0:
        return float("inf")
    return 2.0 * math.sqrt(2.0 * math.log(2.0 * n_components / delta) / M)


def attribute(effects: Dict[str, Sequence[float]], delta: float = 0.05) -> Attribution:
    """effects: component -> M masking replay effects in [-1, 1]."""
    if not effects:
        return Attribution([], {}, None, [], None, True, float("inf"), 0, 0)
    Ms = {len(v) for v in effects.values()}
    if len(Ms) != 1:
        raise ValueError("all components need the same replay budget M")
    M = Ms.pop()
    scores = {}
    for name, vals in effects.items():
        if any(v < -1.0 or v > 1.0 for v in vals):
            raise ValueError(f"effects out of [-1,1] for {name}")
        scores[name] = sum(vals) / M if M else float("nan")
    order = sorted(scores, key=lambda k: scores[k], reverse=True)
    thr = unresolved_threshold(len(scores), M, delta)
    margin = (scores[order[0]] - scores[order[1]]) if len(order) > 1 else None
    unres = (margin is None) or (margin < thr)
    return Attribution(order, scores, (None if unres else order[0]), order[:3],
                       margin, unres, thr, M, len(scores))


def ranking_bound(gamma: float, n_components: int, M: int, tau: float) -> float:
    """exp(-M tau^2/2) + (|U|-1) exp(-M (gamma-tau)^2/2)."""
    return (math.exp(-M * tau * tau / 2.0)
            + (n_components - 1) * math.exp(-M * (gamma - tau) ** 2 / 2.0))


def tau_star_exact(gamma: float, n_components: int, M: int) -> float:
    """Minimizer of the ranking bound over tau in (0, gamma).

    Obtained by direct minimization, NOT by root-finding the stationarity
    condition: that condition can have two roots, and the interior minimizer is
    not the root a naive bisection converges to. The closed form
    gamma/2 - log(|U|-1)/(M gamma) is an approximation and is exposed separately.
    """
    if gamma <= 0 or M <= 0 or n_components < 2:
        return 0.0
    lo, hi = 1e-12, gamma - 1e-12
    # coarse grid to bracket the global minimum, then golden-section refine
    grid = [lo + (hi - lo) * i / 512.0 for i in range(513)]
    best = min(grid, key=lambda t: ranking_bound(gamma, n_components, M, t))
    step = (hi - lo) / 512.0
    a, b = max(lo, best - step), min(hi, best + step)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(200):
        if ranking_bound(gamma, n_components, M, c) < ranking_bound(gamma, n_components, M, d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    return 0.5 * (a + b)


def tau_star_approx(gamma: float, n_components: int, M: int) -> float:
    """First-order approximation only; deployed constants use tau_star_exact."""
    if gamma <= 0 or M <= 0 or n_components < 2:
        return 0.0
    return gamma / 2.0 - math.log(n_components - 1) / (M * gamma)
