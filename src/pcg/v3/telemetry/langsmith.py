"""Optional LangSmith adapter. LANGSMITH_ENABLED=false leaves the core complete.

Local provenance is authoritative. No cloud call is made in this pass, and raw
evidence is never uploaded by default.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

ENABLED = os.environ.get("LANGSMITH_ENABLED", "false").lower() == "true"
UPLOAD_EVIDENCE = os.environ.get("LANGSMITH_UPLOAD_EVIDENCE", "false").lower() == "true"
SENSITIVE = ("evidence", "evidence_hashes", "prompt", "raw_output", "api_key",
             "authorization", "claim")


def scrub(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in payload.items():
        if k.lower() in SENSITIVE and not UPLOAD_EVIDENCE:
            out[k] = "***WITHHELD***"
        else:
            out[k] = v
    return out


class LangSmithAdapter:
    def __init__(self, project: str = "pcg-mas-v3", enabled: Optional[bool] = None):
        self.project = project
        self.enabled = ENABLED if enabled is None else enabled
        self._buffer: List[Dict[str, Any]] = []

    def record(self, kind: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._buffer.append({"kind": kind, **scrub(payload)})

    def flush(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"uploaded": 0, "reason": "LANGSMITH_ENABLED=false"}
        n = len(self._buffer)
        self._buffer.clear()
        return {"uploaded": 0, "buffered": n,
                "reason": "no cloud call during the v3.0 remediation pass"}

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "project": self.project,
                "uploads_evidence_by_default": UPLOAD_EVIDENCE,
                "authoritative_provenance": "local"}
