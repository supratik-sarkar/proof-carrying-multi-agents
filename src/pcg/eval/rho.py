"""
Residual dependence factor rho (Assumption 3 / Eq. 19).

THIS MODULE IS THE STATISTICAL TEETH OF THEOREM 2.

As written, Assumption 3 asserts the existence of rho >= 1 such that
    Pr(cap_i E_i) <= rho^{k-1} prod_i Pr(E_i).
Taken literally this is tautological: one can always set rho = Pr(cap_i E_i) /
prod_i Pr(E_i). For the theorem to have bite we need an *estimator* of rho
that is valid under repeated sampling, together with a confidence upper bound
that makes the bound in Eq. (20) a genuinely testable statement.

We provide:
    - `estimate_rho(...)`  : plug-in ratio estimator on iid trials
    - `rho_ucb(...)`       : Clopper-Pearson / Wilson-based UCB on rho at
                             confidence 1 - alpha
    - `RhoEstimate`        : dataclass carrying the raw stats plus both the
                             plug-in point estimate and the UCB

The UCB rho^UCB is what you plug into the paper's Eq. (20) if you want to
claim a genuinely valid (1 - alpha) upper bound on Pr(false accept) at
redundancy k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RhoEstimate:
    """Results of a rho estimation run.

    Attributes:
        k: redundancy level
        n_trials: number of iid trials in the log
        n_joint_fail: # trials where all k branches failed (cap_i E_i)
        p_marg: plug-in estimate of marginal failure rate (averaged across
                branches under the homogeneous assumption; or per-branch vector
                in the heterogeneous case — here we report the scalar for
                convenience)
        rho_hat: plug-in estimate of rho
        rho_ucb: (1 - alpha) upper confidence bound on rho
        p_joint_ucb: (1 - alpha) upper confidence bound on joint failure prob
    """

    k: int
    n_trials: int
    n_joint_fail: int
    p_marg: float
    rho_hat: float
    rho_ucb: float
    p_joint_ucb: float
    alpha: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "k": self.k,
            "n_trials": self.n_trials,
            "n_joint_fail": self.n_joint_fail,
            "p_marg": self.p_marg,
            "rho_hat": self.rho_hat,
            "rho_ucb": self.rho_ucb,
            "p_joint_ucb": self.p_joint_ucb,
            "alpha": self.alpha,
        }


def _clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """Clopper-Pearson exact upper bound for a binomial proportion.

    P(X >= k | n, p) = alpha when p = upper. Returns 1.0 if k == n (trivial).
    """
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    # Use the Beta quantile relationship:
    #   upper = Beta^{-1}(1 - alpha; k + 1, n - k)
    from scipy.stats import beta
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """Clopper-Pearson exact lower bound for a binomial proportion."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(alpha, k, n - k + 1))


def estimate_rho(
    branch_fail_matrix: np.ndarray,
    alpha: float = 0.05,
) -> RhoEstimate:
    """Estimate rho from a matrix of per-trial branch failure indicators.

    Args:
        branch_fail_matrix: shape (n_trials, k), 0/1 entries. Row t column i
            is 1 iff branch i "failed" (i.e., event E_i occurred) on trial t.
        alpha: 1 - confidence level for the upper confidence bound.

    Returns:
        RhoEstimate with plug-in rho_hat AND rho_ucb.

    Statistical construction (this is the load-bearing part):
        1. Estimate per-branch marginals p_i = mean(branch_fail_matrix[:, i]).
        2. Estimate joint failure prob p_cap = mean(all branches failed).
        3. Plug-in rho_hat = p_cap / prod_i(p_i).
        4. For the UCB, use a one-sided Clopper-Pearson upper bound on p_cap
           and a one-sided Clopper-Pearson LOWER bound on each p_i (pushing
           the ratio numerator up and denominator down — worst case):
               rho_ucb = p_cap_upper / prod_i(p_i_lower).
           We split alpha into (k + 1) pieces via the union bound so that
           the final rho_ucb has simultaneous confidence 1 - alpha.

    This is the quantity you cite alongside Theorem 2 if you want to claim a
    genuine (1 - alpha) upper bound on the false-accept rate.
    """
    m = np.asarray(branch_fail_matrix, dtype=np.int64)
    if m.ndim != 2:
        raise ValueError(f"branch_fail_matrix must be 2D, got shape {m.shape}")
    n_trials, k = m.shape
    if k == 0:
        raise ValueError("need at least one branch")

    # Plug-in marginals
    p_per_branch = m.mean(axis=0)          # shape (k,)
    p_marg = float(p_per_branch.mean())    # scalar summary
    joint_fail_mask = (m.sum(axis=1) == k)
    n_joint = int(joint_fail_mask.sum())
    p_joint = n_joint / n_trials if n_trials else 0.0

    # Plug-in rho
    denom = float(np.prod(p_per_branch)) if np.all(p_per_branch > 0) else 0.0
    rho_hat = (p_joint / denom) if denom > 0 else float("inf")

    # UCB: Bonferroni-split the alpha into k + 1 pieces:
    #   one piece for the numerator upper, k pieces for the denominator lowers.
    alpha_each = alpha / (k + 1)
    p_joint_upper = _clopper_pearson_upper(n_joint, n_trials, alpha_each)
    p_lowers = np.asarray([
        _clopper_pearson_lower(int(m[:, i].sum()), n_trials, alpha_each)
        for i in range(k)
    ])
    denom_lower = float(np.prod(p_lowers)) if np.all(p_lowers > 0) else 0.0
    rho_ucb = (p_joint_upper / denom_lower) if denom_lower > 0 else float("inf")

    return RhoEstimate(
        k=k,
        n_trials=n_trials,
        n_joint_fail=n_joint,
        p_marg=p_marg,
        rho_hat=rho_hat,
        rho_ucb=rho_ucb,
        p_joint_ucb=p_joint_upper,
        alpha=alpha,
    )


def rho_ucb(
    branch_fail_matrix: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Convenience: returns just the rho upper confidence bound."""
    return estimate_rho(branch_fail_matrix, alpha=alpha).rho_ucb


def predicted_false_accept_rate(
    rho_ucb_value: float,
    p_marg_upper: float,
    k: int,
) -> float:
    """Eq. (21) of the paper: rho^{k-1} * eps^k.

    Use `rho_ucb_value` from `rho_ucb(...)` and `p_marg_upper` as a per-branch
    Clopper-Pearson UCB to get a testable upper bound on the false-accept rate
    at redundancy level k.
    """
    if math.isinf(rho_ucb_value):
        return 1.0
    return min(1.0, (rho_ucb_value ** max(0, k - 1)) * (p_marg_upper ** k))
