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
        if not self.allow_exec:
            d = self.fallback.evaluate(request)
            d.backend = "opa_unavailable->local_deterministic"
            d.bundle_sha256 = self.sha
            return d
        proc = subprocess.run(
            [self.binary, "eval", "-d", self.rego_path, "-I", "-f", "json",
             "data.pcg.allow"], input=json.dumps(request), capture_output=True,
            text=True, timeout=30)
        ok = False
        try:
            out = json.loads(proc.stdout)
            ok = bool(out["result"][0]["expressions"][0]["value"])
        except Exception:
            ok = False                      # fail closed on any parse failure
        return PolicyDecision(allowed=ok, rule_matched="rego:data.pcg.allow",
                              reasons=[] if ok else ["denied by rego policy"],
                              bundle_id="pcg", bundle_version="v3.0",
                              bundle_sha256=self.sha, backend=self.name)
