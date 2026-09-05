"""Deterministic in-process policy evaluator (no external dependency)."""
from __future__ import annotations
import json, os
from typing import Any, Dict, List

from ..canon import sha256_obj
from .interface import PolicyDecision

BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "bundles")


def load_bundle(name: str = "default") -> Dict[str, Any]:
    p = os.path.join(BUNDLE_DIR, f"{name}.json")
    with open(p) as fh:
        return json.load(fh)


class LocalPolicyBackend:
    name = "local_deterministic"

    def __init__(self, bundle_name: str = "default"):
        self.bundle = load_bundle(bundle_name)
        self.sha = sha256_obj(self.bundle)

    def evaluate(self, request: Dict[str, Any]) -> PolicyDecision:
        b = self.bundle
        reasons: List[str] = []
        rule = None
        tool = request.get("tool")
        if tool is not None:
            if tool not in b["allowed_tools"]:
                reasons.append(f"tool {tool!r} not in allowlist")
                rule = "tool_allowlist"
        dele = request.get("delegate_to")
        if dele is not None and dele not in b["allowed_delegations"]:
            reasons.append(f"delegation to {dele!r} not permitted")
            rule = rule or "delegation_boundary"
        for k in b.get("required_fields", []):
            if k not in request:
                reasons.append(f"missing required field {k!r}")
                rule = rule or "schema_validation"
        if request.get("verifier_context_shared") is True and b.get("require_verifier_isolation"):
            reasons.append("verifier context shared with prover (isolation clause of Gamma)")
            rule = rule or "verifier_isolation"
        # FAIL CLOSED: unknown request shape is a denial, not an allow.
        if not isinstance(request, dict):
            reasons.append("malformed request")
            rule = "fail_closed"
        return PolicyDecision(allowed=not reasons, rule_matched=rule, reasons=reasons,
                              bundle_id=b["id"], bundle_version=b["version"],
                              bundle_sha256=self.sha, backend=self.name)
