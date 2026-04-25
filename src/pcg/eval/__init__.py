"""
pcg.eval — measurement, statistics, plotting.

Submodules:
    - meter      : per-phase time/token/tool meter (addresses ICML R5 concern)
    - stats      : bootstrap CIs, Wilson intervals, effect sizes
    - rho        : residual-dependence factor estimator with UCB
    - audit      : Theorem 1 decomposition estimators
    - plots      : modern publication-quality plotting style
    - metrics    : standard QA / agent metrics
"""
from __future__ import annotations

from pcg.eval.audit import AuditDecomposition, estimate_audit_decomposition
from pcg.eval.meter import Meter, MeterReport, PhaseTimer
from pcg.eval.metrics import exact_match, f1_score, success_rate
from pcg.eval.rho import RhoEstimate, estimate_rho, rho_ucb
from pcg.eval.stats import bootstrap_ci, hoeffding_ci, wilson_interval

__all__ = [
    "AuditDecomposition",
    "Meter",
    "MeterReport",
    "PhaseTimer",
    "RhoEstimate",
    "bootstrap_ci",
    "estimate_audit_decomposition",
    "estimate_rho",
    "exact_match",
    "f1_score",
    "hoeffding_ci",
    "rho_ucb",
    "success_rate",
    "wilson_interval",
]
