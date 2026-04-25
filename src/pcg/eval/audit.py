"""
Theorem 1 audit-decomposition estimators.

Theorem 1 states
    Pr(accept & (false_claim | unsafe_exec))
        <= Pr(IntFail) + Pr(ReplayFail) + Pr(CheckFail) + Pr(CovGap).

This module provides empirical estimators for each term on the RHS, with
Wilson-score CIs, so that the paper's R1 table can report both the LHS and
each upper-bound component on the same scale.

Mapping of channels to our checker's CheckResult flags:
    - IntFail   : hash mismatch  <=> `integrity_ok == False`
    - ReplayFail: replay failure <=> `replay_ok == False`
    - CheckFail : entailment OR execution-side contract violation
                  <=> `entailment_ok == False` or `execution_ok == False`
    - CovGap    : certificate passed but ground truth disagrees (semantic
                  coverage of R / Gamma was insufficient) — this is the
                  ONLY channel that needs a ground-truth label
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pcg.checker import CheckResult
from pcg.eval.stats import wilson_interval


@dataclass
class AuditDecomposition:
    """Empirical realization of Theorem 1 Eq. (7).

    Each field `p_X` is the Wilson-score estimate of Pr(X); `(lo_X, hi_X)` is
    the 95% CI. `lhs` is the observed P(accept & wrong), i.e., what the union
    bound on the RHS is supposed to upper-bound.
    """

    n: int

    p_int_fail: float
    ci_int_fail: tuple[float, float]

    p_replay_fail: float
    ci_replay_fail: tuple[float, float]

    p_check_fail: float
    ci_check_fail: tuple[float, float]

    p_cov_gap: float
    ci_cov_gap: tuple[float, float]

    # LHS
    lhs_accept_and_wrong: float
    ci_lhs: tuple[float, float]

    # Sum of RHS upper bounds (the Theorem 1 envelope at its own point estimate)
    rhs_union: float

    # Per-item raw flags so downstream code can recompute anything
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "lhs_accept_and_wrong": self.lhs_accept_and_wrong,
            "ci_lhs": list(self.ci_lhs),
            "rhs_union": self.rhs_union,
            "p_int_fail": self.p_int_fail,
            "ci_int_fail": list(self.ci_int_fail),
            "p_replay_fail": self.p_replay_fail,
            "ci_replay_fail": list(self.ci_replay_fail),
            "p_check_fail": self.p_check_fail,
            "ci_check_fail": list(self.ci_check_fail),
            "p_cov_gap": self.p_cov_gap,
            "ci_cov_gap": list(self.ci_cov_gap),
        }


def estimate_audit_decomposition(
    check_results: Sequence[CheckResult],
    ground_truth_correct: Sequence[bool],
    alpha: float = 0.05,
) -> AuditDecomposition:
    """Given N claim-check outcomes and N ground-truth labels, produce the
    empirical version of Theorem 1.

    Args:
        check_results: one CheckResult per claim. `.passed` is the accept
            indicator; the per-channel flags localize failure modes.
        ground_truth_correct: for each claim, was the claim actually correct
            in ground truth? For R1 on HotpotQA, this is whether the claim
            text matches the gold answer.
        alpha: 1 - CI confidence level.

    Channel definitions are in the module docstring.
    """
    n = len(check_results)
    if n != len(ground_truth_correct):
        raise ValueError("check_results and ground_truth_correct must align")

    int_fail = np.zeros(n, dtype=bool)
    replay_fail = np.zeros(n, dtype=bool)
    check_fail = np.zeros(n, dtype=bool)
    cov_gap = np.zeros(n, dtype=bool)
    lhs = np.zeros(n, dtype=bool)

    for i, (r, gt_ok) in enumerate(zip(check_results, ground_truth_correct)):
        int_fail[i] = not r.integrity_ok
        replay_fail[i] = r.integrity_ok and not r.replay_ok
        # CheckFail fires when integrity + replay passed but entailment or exec failed
        check_fail[i] = (
            r.integrity_ok and r.replay_ok
            and not (r.entailment_ok and r.execution_ok)
        )
        # CovGap fires when the certificate PASSED but ground truth disagrees
        cov_gap[i] = r.passed and (not gt_ok)
        # LHS: accept & wrong
        lhs[i] = r.passed and (not gt_ok)

    def _wilson(flags: np.ndarray) -> tuple[float, tuple[float, float]]:
        k = int(flags.sum())
        p_hat, lo, hi = wilson_interval(k, n, alpha=alpha)
        return p_hat, (lo, hi)

    p_int, ci_int = _wilson(int_fail)
    p_rep, ci_rep = _wilson(replay_fail)
    p_chk, ci_chk = _wilson(check_fail)
    p_cov, ci_cov = _wilson(cov_gap)
    p_lhs, ci_lhs = _wilson(lhs)

    return AuditDecomposition(
        n=n,
        p_int_fail=p_int, ci_int_fail=ci_int,
        p_replay_fail=p_rep, ci_replay_fail=ci_rep,
        p_check_fail=p_chk, ci_check_fail=ci_chk,
        p_cov_gap=p_cov, ci_cov_gap=ci_cov,
        lhs_accept_and_wrong=p_lhs, ci_lhs=ci_lhs,
        rhs_union=p_int + p_rep + p_chk + p_cov,
        raw={
            "int_fail": int_fail.tolist(),
            "replay_fail": replay_fail.tolist(),
            "check_fail": check_fail.tolist(),
            "cov_gap": cov_gap.tolist(),
            "lhs": lhs.tolist(),
        },
    )
