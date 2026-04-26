"""
Paired-bootstrap statistical inference for PCG-MAS comparisons.

Every claim in the paper of the form "PCG-MAS achieves X% lower
false-accept rate than baseline Y" should ship with a confidence
interval AND an effect size. This module provides both, with a single
`compare()` entry point that dispatches based on metric type.

Contract:
    - Comparisons are PAIRED whenever possible (same examples through
      both PCG-MAS and the baseline). Paired bootstrap is much tighter
      than unpaired and matters at typical N (~200-1000 examples).
    - Bootstrap CIs use the percentile method by default; BCa available
      via `method="bca"` for skewed distributions.
    - Effect sizes:
        * proportions  → Cohen's h  (small=0.2, medium=0.5, large=0.8)
        * continuous   → Cohen's d  (small=0.2, medium=0.5, large=0.8)
    - Random state is exposed; default seed=0 for reproducibility.

Usage:
    from pcg.eval.bootstrap import compare

    result = compare(
        ours_per_example=ours_safety,        # array of 0/1 per example
        base_per_example=base_safety,        # paired array of 0/1 per example
        metric_kind="proportion",
        n_bootstrap=10_000,
        alpha=0.05,
    )
    print(result.summary())
    # → "PCG-MAS −7.8% ±2.1pp (95% CI [−10.1, −5.6]); Cohen's h = 0.43 (medium)"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


MetricKind = Literal["proportion", "continuous"]


@dataclass
class CompareResult:
    """The full inferential payload for one PCG-vs-baseline comparison."""

    ours_mean: float
    base_mean: float
    delta_mean: float                      # ours - base
    delta_ci: tuple[float, float]
    delta_ci_alpha: float
    bootstrap_n: int
    metric_kind: MetricKind
    effect_size: float                     # Cohen's h or d
    effect_size_label: str                 # "small" | "medium" | "large"
    paired: bool
    n_examples: int
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        kind_short = "h" if self.metric_kind == "proportion" else "d"
        sign = "+" if self.delta_mean >= 0 else "−"
        unit = "pp" if self.metric_kind == "proportion" else ""
        delta_pct = abs(self.delta_mean) * (100 if self.metric_kind == "proportion" else 1)
        ci_low_pct = self.delta_ci[0] * (100 if self.metric_kind == "proportion" else 1)
        ci_high_pct = self.delta_ci[1] * (100 if self.metric_kind == "proportion" else 1)
        ci_pct = (1 - self.delta_ci_alpha) * 100
        return (
            f"PCG-MAS {sign}{delta_pct:.2f}{unit} "
            f"({ci_pct:.0f}% CI [{ci_low_pct:+.2f}, {ci_high_pct:+.2f}]); "
            f"Cohen's {kind_short} = {self.effect_size:.3f} ({self.effect_size_label})"
        )

    def to_dict(self) -> dict:
        return {
            "ours_mean": self.ours_mean,
            "base_mean": self.base_mean,
            "delta_mean": self.delta_mean,
            "delta_ci": list(self.delta_ci),
            "delta_ci_alpha": self.delta_ci_alpha,
            "bootstrap_n": self.bootstrap_n,
            "metric_kind": self.metric_kind,
            "effect_size": self.effect_size,
            "effect_size_label": self.effect_size_label,
            "paired": self.paired,
            "n_examples": self.n_examples,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for the difference between two proportions.

    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))

    Conventional thresholds (absolute value):
        |h| < 0.2  -> negligible
        |h| < 0.5  -> small
        |h| < 0.8  -> medium
        |h| >= 0.8 -> large
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_d(
    x: np.ndarray, y: np.ndarray, paired: bool = True,
) -> float:
    """Cohen's d for continuous samples.

    For paired samples we use the within-subject variant (Cohen's d_z):
        d_z = mean(x - y) / std(x - y, ddof=1)

    For unpaired we use pooled SD (the textbook d).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if paired:
        diff = x - y
        s = diff.std(ddof=1) if diff.size > 1 else 0.0
        if s == 0:
            return 0.0
        return float(diff.mean() / s)
    nx, ny = x.size, y.size
    sx, sy = x.var(ddof=1), y.var(ddof=1)
    pooled = math.sqrt(((nx - 1) * sx + (ny - 1) * sy) / max(nx + ny - 2, 1))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def label_effect(magnitude: float) -> str:
    """Standard Cohen labels for both h and d (same thresholds)."""
    a = abs(magnitude)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    ours: np.ndarray,
    base: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    statistic: callable = np.mean,
    method: Literal["percentile", "bca"] = "percentile",
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Paired bootstrap CI for a difference statistic(ours) - statistic(base).

    Bootstrap resamples INDICES, then evaluates the statistic on the
    paired-resampled vectors. This preserves correlation structure and
    is the right thing for "same examples, two methods" comparisons.

    Args:
        ours, base: 1-D arrays of equal length.
        n_bootstrap: number of resamples (10k is safe; 1k for fast iter).
        alpha: 0.05 → 95% CI.
        statistic: any reduction np.mean → mean, np.median, etc.
        method: "percentile" (default) or "bca" for bias-corrected accelerated.
        seed: PRNG seed for reproducibility.

    Returns:
        (point_estimate, (ci_low, ci_high))
    """
    ours = np.asarray(ours, dtype=float)
    base = np.asarray(base, dtype=float)
    if ours.shape != base.shape:
        raise ValueError(
            f"paired bootstrap needs equal-length arrays; "
            f"got ours={ours.shape}, base={base.shape}"
        )
    n = ours.size
    if n == 0:
        return 0.0, (0.0, 0.0)

    rng = np.random.default_rng(seed)
    point = float(statistic(ours) - statistic(base))

    # Vectorized resampling: shape (n_bootstrap, n) of indices
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Statistic must be computable on a 1D slice
        deltas[b] = statistic(ours[idx[b]]) - statistic(base[idx[b]])

    if method == "bca":
        ci = _bca_ci(deltas, point, ours, base, statistic, alpha, rng)
    else:
        lo = float(np.quantile(deltas, alpha / 2))
        hi = float(np.quantile(deltas, 1 - alpha / 2))
        ci = (lo, hi)
    return point, ci


def _bca_ci(
    deltas: np.ndarray, point: float,
    ours: np.ndarray, base: np.ndarray,
    statistic: callable, alpha: float, rng: np.random.Generator,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI.

    Adjusts for skew and bias in the bootstrap distribution. Worth using
    when the metric is bounded (proportions near 0 or 1) or skewed."""
    # Bias-correction factor z0
    p_below = float((deltas < point).mean())
    p_below = min(max(p_below, 1e-6), 1 - 1e-6)
    z0 = _norm_ppf(p_below)

    # Acceleration via jackknife
    n = ours.size
    jack_deltas = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        jack_deltas[i] = statistic(ours[mask]) - statistic(base[mask])
    jack_mean = jack_deltas.mean()
    num = ((jack_mean - jack_deltas) ** 3).sum()
    den = 6.0 * ((jack_mean - jack_deltas) ** 2).sum() ** 1.5
    a = num / den if den != 0 else 0.0

    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)
    alpha_lo = _norm_cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    alpha_hi = _norm_cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    lo = float(np.quantile(deltas, alpha_lo))
    hi = float(np.quantile(deltas, alpha_hi))
    return (lo, hi)


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF — Beasley-Springer-Moro approximation.

    Avoids the scipy dependency. Accurate to ~1e-7 on (0, 1)."""
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Top-level compare()
# ---------------------------------------------------------------------------


def compare(
    *,
    ours_per_example: np.ndarray,
    base_per_example: np.ndarray,
    metric_kind: MetricKind = "proportion",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    method: Literal["percentile", "bca"] = "percentile",
    paired: bool = True,
    seed: int = 0,
) -> CompareResult:
    """One-stop comparison: paired bootstrap CI + appropriate effect size.

    Always computes BOTH (a) the bootstrap CI on the difference and
    (b) the appropriate effect size (Cohen's h for proportions, Cohen's
    d for continuous). Effect sizes are scale-free; CIs are in the
    metric's native units.
    """
    ours = np.asarray(ours_per_example, dtype=float)
    base = np.asarray(base_per_example, dtype=float)
    if not paired:
        raise NotImplementedError(
            "Unpaired bootstrap not implemented; pad the shorter array "
            "and pass paired=True instead."
        )

    point, ci = paired_bootstrap_ci(
        ours, base,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        method=method,
        seed=seed,
    )

    if metric_kind == "proportion":
        eff = cohens_h(float(ours.mean()), float(base.mean()))
    else:
        eff = cohens_d(ours, base, paired=True)

    return CompareResult(
        ours_mean=float(ours.mean()),
        base_mean=float(base.mean()),
        delta_mean=point,
        delta_ci=ci,
        delta_ci_alpha=alpha,
        bootstrap_n=n_bootstrap,
        metric_kind=metric_kind,
        effect_size=eff,
        effect_size_label=label_effect(eff),
        paired=paired,
        n_examples=int(ours.size),
    )
