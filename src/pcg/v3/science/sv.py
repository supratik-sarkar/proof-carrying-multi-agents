"""Exact selectivity/verification decomposition.

    S = mean_[N](l_nc) - mean_[A](l_nc)
    V = mean_[A](l_nc - l_pcg)
    Delta = S + V                                (algebraic identity, exact)
    S = (1 - alpha) * (mean_[Ac](l_nc) - mean_[A](l_nc)),  alpha = |A|/N
    |S| <= (1 - alpha) * range(l_nc)

At full coverage S == 0. At matched coverage only alpha is equalised; S may
still differ because systems refuse different examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class SVResult:
    N: int
    N_acc: int
    alpha: float
    S: Optional[float]
    V: Optional[float]
    delta: Optional[float]
    identity_residual: Optional[float]
    s_bound: Optional[float]

    def to_dict(self) -> dict:
        return vars(self)


def sv_decomposition(loss_nocert: Sequence[Optional[float]],
                     loss_pcg: Sequence[Optional[float]],
                     answered: Sequence[bool]) -> SVResult:
    """Per-example decomposition. Refused rows must carry loss_pcg = None."""
    n = len(loss_nocert)
    if len(loss_pcg) != n or len(answered) != n:
        raise ValueError("ragged inputs")
    if n == 0:
        return SVResult(0, 0, 0.0, None, None, None, None, None)
    if any(x is None for x in loss_nocert):
        raise ValueError("loss_nocert must be defined on every example")

    A = [i for i in range(n) if answered[i]]
    na = len(A)
    alpha = na / n
    mean_all = sum(loss_nocert) / n
    if na == 0:
        return SVResult(n, 0, 0.0, None, None, None, None, None)
    if any(loss_pcg[i] is None for i in A):
        raise ValueError("answered examples must carry a PCG loss")

    mean_A_nc = sum(loss_nocert[i] for i in A) / na
    mean_A_pcg = sum(loss_pcg[i] for i in A) / na
    S = mean_all - mean_A_nc
    V = mean_A_nc - mean_A_pcg
    delta = mean_all - mean_A_pcg
    rng = max(loss_nocert) - min(loss_nocert)
    return SVResult(n, na, alpha, S, V, delta, abs((S + V) - delta), (1 - alpha) * rng)


def assert_identity(res: SVResult, tol: float = 1e-12) -> None:
    if res.identity_residual is None:
        return
    if res.identity_residual > tol:
        raise AssertionError(
            f"S+V != Delta (residual {res.identity_residual:.3e} > {tol:.0e}); "
            "this indicates a metric-implementation defect, not sampling noise")
