"""Typed V_Gamma decision records with full correlation.

Every AVAILABLE decision links: decision_id <-> trace_id/span_id <-> policy input
hash <-> bundle revision/hash <-> execution context <-> certificate.
Unavailable or malformed evaluation is INDETERMINATE -- never ALLOW, and never a
measured violation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..canon import sha256_obj
from ..exec.policy_status import PolicyEvalStatus, PolicyOutcome, RejectionReason


@dataclass
class PolicyDecisionRecord:
    decision_id: str
    policy_eval_status: str
    v_gamma: Optional[bool]
    matched_rules: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    policy_path: Optional[str] = None
    bundle_id: Optional[str] = None
    bundle_revision: Optional[str] = None
    bundle_sha256: Optional[str] = None
    policy_input_hash: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    execution_context_id: Optional[str] = None
    certificate_root: Optional[str] = None
    fail_closed: bool = True
    timestamp_ms: float = 0.0
    backend: str = "local_deterministic"

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)

    @property
    def outcome(self) -> PolicyOutcome:
        return PolicyOutcome(PolicyEvalStatus(self.policy_eval_status), self.v_gamma,
                             self.decision_id, "; ".join(self.reasons))

    @property
    def rejection_reason(self) -> str:
        return self.outcome.rejection_reason.value


def make_decision(policy_input: Dict[str, Any], allowed: Optional[bool], available: bool,
                  matched_rules: Optional[List[str]] = None,
                  reasons: Optional[List[str]] = None,
                  bundle: Optional[Dict[str, str]] = None,
                  trace_id: Optional[str] = None, span_id: Optional[str] = None,
                  execution_context_id: Optional[str] = None,
                  certificate_root: Optional[str] = None,
                  policy_path: str = "data.pcg.allow",
                  backend: str = "local_deterministic") -> PolicyDecisionRecord:
    b = bundle or {}
    pih = sha256_obj(policy_input)          # committed fields only, by construction
    did = "pd-" + sha256_obj({"in": pih, "b": b.get("sha256"), "p": policy_path,
                              "t": trace_id, "s": span_id})[:24]
    if not available or allowed is None:
        return PolicyDecisionRecord(
            decision_id=did, policy_eval_status=PolicyEvalStatus.INDETERMINATE.value,
            v_gamma=None, matched_rules=[], policy_path=policy_path,
            reasons=(reasons or ["policy evaluation unavailable or malformed"]),
            bundle_id=b.get("id"), bundle_revision=b.get("version"),
            bundle_sha256=b.get("sha256"), policy_input_hash=pih,
            trace_id=trace_id, span_id=span_id,
            execution_context_id=execution_context_id, certificate_root=certificate_root,
            timestamp_ms=time.time() * 1000.0, backend=backend)
    return PolicyDecisionRecord(
        decision_id=did, policy_eval_status=PolicyEvalStatus.AVAILABLE.value,
        v_gamma=bool(allowed), matched_rules=matched_rules or [],
        reasons=reasons or [], policy_path=policy_path,
        bundle_id=b.get("id"), bundle_revision=b.get("version"),
        bundle_sha256=b.get("sha256"), policy_input_hash=pih,
        trace_id=trace_id, span_id=span_id,
        execution_context_id=execution_context_id, certificate_root=certificate_root,
        timestamp_ms=time.time() * 1000.0, backend=backend)
