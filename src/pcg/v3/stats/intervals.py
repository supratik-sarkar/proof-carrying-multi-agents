"""Binomial confidence bounds without a hard SciPy dependency.

Clopper-Pearson uses Beta quantiles. SciPy is used when present; otherwise an
exact-enough regularized incomplete beta (Lentz continued fraction) plus
bisection is used, so the scientific core runs on a bare interpreter.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

try:                       # optional fast path
    from scipy.stats import beta as _scipy_beta   # type: ignore
    _HAVE_SCIPY = True
except Exception:          # pragma: no cover - exercised on minimal envs
    _HAVE_SCIPY = False


# ---------------------------------------------------------------- incomplete beta
def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                          + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Beta quantile by bisection on the regularized incomplete beta."""
    if a <= 0 or b <= 0:
        raise ValueError("beta_ppf requires a, b > 0")
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- Clopper-Pearson
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Two-sided Clopper-Pearson interval for k successes in n trials.

    Returns (lower, upper). NOTE the exact boundary behaviour that v3.0 relies on:
    k == 0 gives lower == 0.0 exactly, which is what makes the dependence gate
    fail closed (and is why an evidence floor is required -- see science.dependence).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= k <= n):
        raise ValueError("k must satisfy 0 <= k <= n")
    lo = 0.0 if k == 0 else _ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else _ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return lo, hi


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided upper limit (used for joint co-failure rates)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 if k == n else _ppf(1.0 - alpha, k + 1, n - k)


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided lower limit (used for marginal branch failure rates)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 0.0 if k == 0 else _ppf(alpha, k, n - k + 1)


def _ppf(q: float, a: float, b: float) -> float:
    if _HAVE_SCIPY:
        return float(_scipy_beta.ppf(q, a, b))
    return beta_ppf(q, a, b)


# ---------------------------------------------------------------- Hoeffding
def hoeffding_halfwidth(n: int, delta: float, n_tests: int = 1) -> float:
    """One-sided Hoeffding half-width sqrt(log(n_tests/delta) / (2n))."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt(math.log(n_tests / delta) / (2.0 * n))


def wilson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Wilson lower bound; reported only as a secondary diagnostic."""
    if n <= 0:
        return 0.0
    z = _z(1.0 - alpha)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - rad) / d)


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
