"""
Core statistics: distribution-free and parametric CIs used throughout the paper.

    - `bootstrap_ci`      : percentile bootstrap + BCa (bias-corrected)
    - `hoeffding_ci`      : for bounded random variables, distribution-free
    - `wilson_interval`   : binomial proportion CI, better than Wald on small n
    - `kendall_tau`       : ranking stability (Appendix C.8, Eq. 58)
    - `top_k_overlap`     : Appendix C.8 Eq. 59
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    n_boot: int
    method: str


def bootstrap_ci(
    data: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = lambda a: float(a.mean()),
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "percentile",
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap confidence interval for a scalar statistic.

    method:
        "percentile": Efron percentile method (fast, usually fine for means)
        "bca":        Bias-corrected and accelerated (uses jackknife for a)
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return BootstrapCI(math.nan, math.nan, math.nan, 0, method)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.asarray([statistic(arr[i]) for i in idx])
    theta_hat = statistic(arr)

    if method == "percentile":
        lo = float(np.quantile(boots, alpha / 2))
        hi = float(np.quantile(boots, 1 - alpha / 2))
        return BootstrapCI(theta_hat, lo, hi, n_boot, "percentile")

    if method == "bca":
        # Bias-correction
        p0 = float((boots < theta_hat).mean())
        if p0 in (0.0, 1.0):
            # Fall back to percentile if the bias correction is undefined
            return bootstrap_ci(arr, statistic, n_boot, alpha, "percentile", seed)
        from scipy.stats import norm
        z0 = float(norm.ppf(p0))
        # Acceleration via jackknife
        jk = np.asarray([statistic(np.delete(arr, i)) for i in range(n)])
        jk_mean = jk.mean()
        num = ((jk_mean - jk) ** 3).sum()
        den = 6.0 * (((jk_mean - jk) ** 2).sum() ** 1.5)
        a = num / den if den > 0 else 0.0
        z_alpha1 = float(norm.ppf(alpha / 2))
        z_alpha2 = float(norm.ppf(1 - alpha / 2))
        alpha1 = float(norm.cdf(z0 + (z0 + z_alpha1) / (1 - a * (z0 + z_alpha1))))
        alpha2 = float(norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2))))
        lo = float(np.quantile(boots, alpha1))
        hi = float(np.quantile(boots, alpha2))
        return BootstrapCI(theta_hat, lo, hi, n_boot, "bca")

    raise ValueError(f"Unknown bootstrap method: {method}")


# -----------------------------------------------------------------------------
# Hoeffding CI
# -----------------------------------------------------------------------------


def hoeffding_ci(
    samples: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    lo_bound: float = 0.0,
    hi_bound: float = 1.0,
) -> tuple[float, float, float]:
    """Hoeffding's (distribution-free) CI for the mean of bounded samples.

    For samples in [lo_bound, hi_bound] the (1-alpha) CI for the mean is
        mean_hat +/- (hi - lo) * sqrt(log(2/alpha) / (2n))

    Returns (mean_hat, lower, upper).
    """
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    m = float(arr.mean())
    rng = hi_bound - lo_bound
    hw = rng * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    return (m, m - hw, m + hw)


# -----------------------------------------------------------------------------
# Wilson interval
# -----------------------------------------------------------------------------


def wilson_interval(
    k: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Wilson-score CI for a binomial proportion.

    Recommended over Wald CIs, especially when k is near 0 or n. Returns
    (p_hat, lower, upper).
    """
    if n == 0:
        return (math.nan, 0.0, 1.0)
    # z for two-sided alpha
    from scipy.stats import norm
    z = float(norm.ppf(1 - alpha / 2))
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (p_hat, max(0.0, center - half), min(1.0, center + half))


# -----------------------------------------------------------------------------
# Ranking metrics (Appendix C.8)
# -----------------------------------------------------------------------------


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float:
    """Kendall's tau for two rankings over the same item set."""
    from scipy.stats import kendalltau
    rank_a = {x: i for i, x in enumerate(a)}
    rank_b = {x: i for i, x in enumerate(b)}
    common = [x for x in a if x in rank_b]
    if len(common) < 2:
        return math.nan
    tau, _ = kendalltau(
        [rank_a[x] for x in common],
        [rank_b[x] for x in common],
    )
    return float(tau)


def top_k_overlap(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """|TopK(a) cap TopK(b)| / k."""
    ta = set(a[:k])
    tb = set(b[:k])
    return len(ta & tb) / max(1, k)
