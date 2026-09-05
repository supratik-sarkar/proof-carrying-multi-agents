"""V_Gamma policy backend interface.

OPA is ONE implementation backend for V_Gamma. It does not redefine the
mathematical meaning of V_Gamma, and the local deterministic evaluator must work
with OPA absent. Every decision carries the policy bundle version and SHA-256.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class PolicyDecision:
    allowed: bool
    rule_matched: Optional[str]
    reasons: List[str]
    bundle_id: str
    bundle_version: str
    bundle_sha256: str
    backend: str
    fail_closed: bool = True

    def to_dict(self) -> dict:
        return vars(self)


class PolicyBackend(Protocol):
    name: str
    def evaluate(self, request: Dict[str, Any]) -> PolicyDecision: ...
