"""Certificate-level shift monitoring.

Acceptance is measurable w.r.t. the certificate sigma-algebra F_Z, so

    |P_t(E) - P_cal(E)| <= TV(P_t|F_Z, P_cal|F_Z) <= TV(P_t, P_cal)

and for the Bayes-optimal certificate-summary classifier, 2a* - 1 == TV|F_Z.

CRITICAL SCOPE: a realizable classifier gives only the LOWER bound 2a - 1.
D_alarm is therefore a GATE, never substituted for a valid upper bound D_bar.
Semantic falsehood and eps_src are not generally F_Z-measurable, so this does
NOT convert certificate-level monitoring into a semantic bad-accept guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..stats.intervals import _z


@dataclass
class ShiftAlarm:
    balanced_accuracy: float
    a_lcb: float
    d_alarm: float
    n_cal: int
    n_dep: int
    triggered: bool
    d_bar: Optional[float] = None
    note: str = ("D_alarm is a one-sided lower bound on restricted TV: large proves shift, "
                 "small does NOT certify absence. Never substitute for D_bar.")

    def to_dict(self) -> dict:
        return vars(self)


def balanced_accuracy_lcb(a_hat: float, n_cal: int, n_dep: int, alpha: float = 0.05) -> float:
    """Pre-specified lower confidence bound on held-out balanced accuracy."""
    if n_cal <= 0 or n_dep <= 0:
        return 0.0
    z = _z(1.0 - alpha)
    var = 0.25 * (a_hat * (1 - a_hat)) * (1.0 / n_cal + 1.0 / n_dep)
    return max(0.0, min(1.0, a_hat - z * (var ** 0.5)))


def shift_alarm(a_hat: float, n_cal: int, n_dep: int, threshold: float = 0.10,
                alpha: float = 0.05, d_bar: Optional[float] = None) -> ShiftAlarm:
    """D_alarm = max{0, 2*a_LCB - 1}."""
    lcb = balanced_accuracy_lcb(a_hat, n_cal, n_dep, alpha)
    d = max(0.0, 2.0 * lcb - 1.0)
    return ShiftAlarm(a_hat, lcb, d, n_cal, n_dep, d > threshold, d_bar)


def transfer_bound(p_cal_event: float, d_bar: Optional[float]) -> Optional[float]:
    """P_t(E) <= P_cal(E) + D_bar. Returns None unless an INDEPENDENTLY VALID
    upper bound on restricted TV is supplied. D_alarm is not accepted here."""
    if d_bar is None:
        return None
    return min(1.0, p_cal_event + d_bar)
