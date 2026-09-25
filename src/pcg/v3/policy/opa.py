"""OPA/Rego adapter. Not contacted during the remediation pass.

If the `opa` binary is unavailable the adapter reports BLOCKED and the caller
falls back to the local deterministic evaluator; the scientific meaning of
V_Gamma is unchanged either way.
"""
from __future__ import annotations
import json, os, shutil, subprocess
from typing import Any, Dict, Optional

from ..canon import sha256_file, sha256_obj
from .interface import PolicyDecision
from .local import BUNDLE_DIR, LocalPolicyBackend


class OPAPolicyBackend:
    name = "opa"

    def __init__(self, bundle_name: str = "default", binary: Optional[str] = None,
                 allow_exec: bool = False):
        self.binary = binary or shutil.which("opa")
        self.allow_exec = allow_exec and bool(self.binary)
        self.rego_path = os.path.join(BUNDLE_DIR, f"{bundle_name}.rego")
        self.fallback = LocalPolicyBackend(bundle_name)
        self.sha = (sha256_file(self.rego_path) if os.path.exists(self.rego_path)
                    else self.fallback.sha)

    @property
    def available(self) -> bool:
        return bool(self.binary) and os.path.exists(self.rego_path)

    def status(self) -> Dict[str, Any]:
        return {"binary": self.binary, "rego_present": os.path.exists(self.rego_path),
                "available": self.available, "exec_enabled": self.allow_exec,
                "bundle_sha256": self.sha}

    def evaluate(self, request: Dict[str, Any]) -> PolicyDecision:
        if not self.allow_exec or not self.binary or not os.path.exists(self.rego_path):
            d = self.fallback.evaluate(request)
            d.backend = "local_deterministic_mirror"
            d.bundle_sha256 = self.sha
            return d
        
        cmd = [self.binary, "eval", "--v0-compatible", "-d", self.rego_path, "-I", "-f", "json", "data.pcg.allow"]
        try:
            proc = subprocess.run(
                cmd, input=json.dumps(request), capture_output=True,
                text=True, timeout=30
            )
        except Exception as exc:
            return PolicyDecision(allowed=False, rule_matched="opa:error",
                                  reasons=[f"opa execution exception: {exc}"],
                                  bundle_id="pcg", bundle_version="v3.2",
                                  bundle_sha256=self.sha, backend=self.name)

        ok = False
        reasons = []
        rule = "rego:data.pcg.allow"
        try:
            out = json.loads(proc.stdout)
            ok = bool(out["result"][0]["expressions"][0]["value"])
            if not ok:
                reasons.append("denied by rego policy")
        except Exception as exc:
            ok = False                      # fail closed on any parse failure
            reasons.append(f"parse error from opa output: {exc}")
            rule = "opa:parse_error"

        return PolicyDecision(allowed=ok, rule_matched=rule,
                              reasons=reasons, bundle_id="pcg", bundle_version="v3.2",
                              bundle_sha256=self.sha, backend=self.name)

    def evaluate_record(self, request: Dict[str, Any], trace_id: Optional[str] = None,
                        span_id: Optional[str] = None,
                        execution_context_id: Optional[str] = None) -> Any:
        from .decision import make_decision
        if not self.available:
            return make_decision(
                policy_input=request, allowed=None, available=False,
                reasons=["OPA runtime binary or bundle unavailable"],
                bundle={"id": "pcg", "version": "v3.2", "sha256": self.sha},
                trace_id=trace_id, span_id=span_id, execution_context_id=execution_context_id,
                backend=self.name
            )
        dec = self.evaluate(request)
        return make_decision(
            policy_input=request, allowed=dec.allowed, available=True,
            matched_rules=[dec.rule_matched] if dec.rule_matched else [],
            reasons=dec.reasons,
            bundle={"id": dec.bundle_id, "version": dec.bundle_version, "sha256": dec.bundle_sha256},
            trace_id=trace_id, span_id=span_id, execution_context_id=execution_context_id,
            backend=dec.backend
        )
