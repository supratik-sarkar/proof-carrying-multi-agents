"""
Demo-runtime implementation of the PCG-MAS paper algorithm.

This module implements the same public semantics as the paper implementation in:
    src/pcg/eval/bootstrap.py

It is intentionally written against the lightweight demo schema in
app/pcg_glue/schemas.py so the Hugging Face Space remains self-contained
and does not require the full R1-R5 experiment harness objects such as
GroundingCertificate, Checker, or AgenticRuntimeGraph.

Lifted algorithms (paper equations preserved):
    - paired-bootstrap CI on a per-example statistic
    - Wilson-score CI for pass-rate proportions (used when n is tiny)
    - cohens_h / cohens_d effect sizes
    - label_effect (negligible / small / medium / large)

Demo usage: for each AuditChannel (V_I, V_R, V_D, V_Ch, V_Cov), we compute
the pass-rate across this run's claims and a 95% CI on that pass-rate.
With typical demo N (1-10 claims per run) the paired-bootstrap CI degenerates,
so we use Wilson by default and offer paired-bootstrap when N >= 20.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from pcg_glue.schemas import (
    AuditEnvelope, ChannelName, ChannelState,
    ClaimCertificate,
)


# =============================================================================
# Wilson-score CI for a single proportion (tight at small N)
# =============================================================================

def wilson_ci(
    successes: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Wilson-score CI for a binomial proportion.

    Returns (p_hat, ci_low, ci_high). At n=0, returns (0, 0, 1).
    """
    if n <= 0:
        return 0.0, 0.0, 1.0
    p_hat = successes / n
    z = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}.get(alpha, 1.96)
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return p_hat, max(0.0, center - half), min(1.0, center + half)


# =============================================================================
# Paired bootstrap CI (used when N >= 20)
# =============================================================================

def paired_bootstrap_ci(
    samples: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Paired bootstrap CI on the mean of `samples` (0/1 array).

    Bootstrap resamples INDICES, then takes the mean. Percentile CI.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    point = float(samples.mean())
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = samples[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return point, (lo, hi)


# =============================================================================
# Top-level: build per-channel envelopes from a list of ClaimCertificates
# =============================================================================

def compute_envelopes(
    claim_certs: list[ClaimCertificate],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> list[AuditEnvelope]:
    """For each of the 5 channels, compute pass-rate + CI across this run's claims.

    Uses Wilson when n < 20 (tight at tiny N) and paired-bootstrap otherwise.
    Returns a list of AuditEnvelope in canonical channel order:
        V_I, V_R, V_D, V_Ch, V_Cov
    """
    if not claim_certs:
        return []

    canonical = [
        ChannelName.V_I,
        ChannelName.V_R,
        ChannelName.V_D,
        ChannelName.V_Ch,
        ChannelName.V_Cov,
    ]

    envelopes: list[AuditEnvelope] = []
    n = len(claim_certs)
    for ch in canonical:
        # 1 if claim's channel verdict is PASS, else 0. SKIP counts as 0
        # because we want pass-rate, not "non-failure" rate.
        flags = np.array([
            1 if (cc.channels.get(ch) is not None
                  and cc.channels[ch].state == ChannelState.PASS) else 0
            for cc in claim_certs
        ], dtype=float)

        if n >= 20:
            pass_rate, (lo, hi) = paired_bootstrap_ci(
                flags, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed,
            )
            method = "paired_bootstrap"
        else:
            successes = int(flags.sum())
            pass_rate, lo, hi = wilson_ci(successes, n, alpha=alpha)
            method = "wilson"

        envelopes.append(AuditEnvelope(
            channel=ch,
            pass_rate=round(pass_rate, 4),
            ci_low=round(lo, 4),
            ci_high=round(hi, 4),
            n_samples=n,
            n_bootstrap=n_bootstrap if method == "paired_bootstrap" else 0,
            method=method,
        ))
    return envelopes


# =============================================================================
# Effect sizes (kept for parity with paper API; unused by the live demo
# but valuable for the alignment test in tests/test_demo_runtime_alignment.py)
# =============================================================================

def cohens_h(p1: float, p2: float) -> float:
    """h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))."""
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_d(x: np.ndarray, y: np.ndarray, paired: bool = True) -> float:
    """Cohen's d (paired: d_z within-subject; unpaired: pooled SD)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if paired:
        diff = x - y
        s = diff.std(ddof=1) if diff.size > 1 else 0.0
        return float(diff.mean() / s) if s else 0.0
    nx, ny = x.size, y.size
    sx, sy = x.var(ddof=1), y.var(ddof=1)
    pooled = math.sqrt(((nx - 1) * sx + (ny - 1) * sy) / max(nx + ny - 2, 1))
    return float((x.mean() - y.mean()) / pooled) if pooled else 0.0


def label_effect(magnitude: float) -> str:
    a = abs(magnitude)
    if a < 0.2:  return "negligible"
    if a < 0.5:  return "small"
    if a < 0.8:  return "medium"
    return "large"


__all__ = [
    "compute_envelopes",
    "wilson_ci",
    "paired_bootstrap_ci",
    "cohens_h",
    "cohens_d",
    "label_effect",
]
