"""Policy evaluation availability, separated from policy violation.

The mathematical conjunct stays Boolean, exactly as locked:

    policy_eval_status in {AVAILABLE, INDETERMINATE}
    V_Gamma            in {True, False, None}

    AVAILABLE + allowed   -> V_Gamma = True
    AVAILABLE + violated  -> V_Gamma = False
    unavailable/malformed/timeout/infrastructure failure
                          -> policy_eval_status = INDETERMINATE, V_Gamma = None

Operationally acceptance FAILS CLOSED when a required evaluation is
indeterminate. Statistically an indeterminate observation is NOT a violation:

    policy_violation_rate            = #{V_Gamma is False} / #{V_Gamma in {True,False}}
    policy_evaluation_unavailability = #{status == INDETERMINATE} / N

`INDETERMINATE` must never be reportable as `REJECT_DUE_TO_POLICY_VIOLATION`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional


class PolicyEvalStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    INDETERMINATE = "INDETERMINATE"


class RejectionReason(str, Enum):
    POLICY_VIOLATION = "REJECT_DUE_TO_POLICY_VIOLATION"
    POLICY_INDETERMINATE = "REJECT_DUE_TO_POLICY_INDETERMINATE"
    OTHER_CONJUNCT = "REJECT_DUE_TO_OTHER_CONJUNCT"
    NONE = "NONE"


@dataclass
class PolicyOutcome:
    status: PolicyEvalStatus
    v_gamma: Optional[bool]
    decision_id: Optional[str] = None
    reason: str = ""

    @property
    def rejection_reason(self) -> RejectionReason:
        if self.status is PolicyEvalStatus.INDETERMINATE:
            return RejectionReason.POLICY_INDETERMINATE   # never *_VIOLATION
        if self.v_gamma is False:
            return RejectionReason.POLICY_VIOLATION
        return RejectionReason.NONE

    @property
    def fails_closed(self) -> bool:
        """Acceptance requires V_Gamma is True; None and False both block."""
        return self.v_gamma is not True

    def to_dict(self) -> Dict[str, Any]:
        return {"policy_eval_status": self.status.value, "v_gamma": self.v_gamma,
                "decision_id": self.decision_id, "reason": self.reason,
                "rejection_reason": self.rejection_reason.value}


def from_backend(allowed: Optional[bool], available: bool,
                 decision_id: Optional[str] = None, reason: str = "") -> PolicyOutcome:
    if not available or allowed is None:
        return PolicyOutcome(PolicyEvalStatus.INDETERMINATE, None, decision_id,
                             reason or "policy evaluation unavailable or malformed")
    return PolicyOutcome(PolicyEvalStatus.AVAILABLE, bool(allowed), decision_id, reason)


def policy_rates(records: Iterable[dict]) -> Dict[str, Optional[float]]:
    """Violation rate excludes indeterminate; unavailability is reported separately."""
    n = viol = defined = indet = 0
    for r in records:
        n += 1
        st = r.get("policy_eval_status")
        vg = r.get("v_gamma")
        if st == PolicyEvalStatus.INDETERMINATE.value or vg is None:
            indet += 1
            continue
        defined += 1
        if vg is False:
            viol += 1
    return {
        "N": n,
        "policy_violation_rate": (viol / defined) if defined else None,   # None, never 0.0
        "policy_violation_numerator": viol,
        "policy_violation_denominator": defined,
        "policy_evaluation_unavailability_rate": (indet / n) if n else None,
        "indeterminate_count": indet,
    }
