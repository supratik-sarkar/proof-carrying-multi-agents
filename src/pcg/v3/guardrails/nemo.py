"""Optional NeMo Guardrails boundary-control adapter.

It may NOT silently rewrite scientific inputs, may NOT become the hidden
implementation of PCG, and every intervention is logged and provenance-tagged.
Core PCG execution works with NeMo absent.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ENABLED = os.environ.get("NEMO_GUARDRAILS_ENABLED", "false").lower() == "true"

try:
    import nemoguardrails  # type: ignore
    HAVE_NEMO = True
except Exception:
    HAVE_NEMO = False


@dataclass
class Intervention:
    stage: str
    rule: str
    action: str            # "flag" | "block"
    original_sha256: str
    note: str = "boundary control only; scientific input is never silently rewritten"

    def to_dict(self) -> dict:
        return vars(self)


class NeMoAdapter:
    name = "nemo_guardrails"

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = (ENABLED if enabled is None else enabled) and HAVE_NEMO
        self.log: List[Intervention] = []

    def check_input(self, text: str) -> Dict[str, Any]:
        from ..canon import sha256_text
        if not self.enabled:
            return {"applied": False, "reason": "adapter disabled or NeMo not installed",
                    "text": text}
        blocked = any(m in text.lower() for m in ("ignore previous instructions",
                                                  "disregard the system prompt"))
        if blocked:
            self.log.append(Intervention("input", "prompt_injection_pattern", "flag",
                                         sha256_text(text)))
        return {"applied": True, "flagged": blocked, "text": text,
                "interventions": [i.to_dict() for i in self.log]}

    def status(self) -> Dict[str, Any]:
        return {"installed": HAVE_NEMO, "enabled": self.enabled,
                "rewrites_inputs": False, "interventions_logged": len(self.log)}
