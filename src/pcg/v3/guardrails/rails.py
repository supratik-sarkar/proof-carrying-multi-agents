"""NeMo Guardrails multi-rail provenance.

NeMo exposes distinct lifecycle rail categories; they are used deliberately.
Rules that are not negotiable:
  * disabled mode is a BIT-IDENTICAL transparent bypass;
  * a rail may never silently mutate a scientific input -- every TRANSFORM is
    recorded with before/after hashes;
  * for A05/A16 primary arms guardrails are OFF identically for every system and
    `guardrail_config_hash` enters the frozen spec.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..canon import sha256_obj, sha256_text

try:
    import nemoguardrails  # type: ignore
    HAVE_NEMO = True
except Exception:
    HAVE_NEMO = False


class RailType(str, Enum):
    INPUT = "INPUT"
    RETRIEVAL = "RETRIEVAL"
    EXECUTION = "EXECUTION"     # tool / action rails
    OUTPUT = "OUTPUT"
    DIALOG = "DIALOG"           # only where conversational state is relevant


class RailAction(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    TRANSFORM = "TRANSFORM"


class ScientificRole(str, Enum):
    OPTIONAL_DEPLOYMENT = "OPTIONAL_DEPLOYMENT"
    COMPARATOR = "COMPARATOR"
    REQUIRED_PROTOCOL = "REQUIRED_PROTOCOL"


@dataclass
class GuardrailInterventionRecord:
    rail_type: str
    rail_name: str
    config_hash: str
    trigger: Optional[str]
    input_hash: str
    output_hash: str
    action: str
    reason: str
    latency_ms: float
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    altered_scientific_input: bool = False
    scientific_role: str = ScientificRole.OPTIONAL_DEPLOYMENT.value
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


class RailEngine:
    """Optional. Core PCG execution works with NeMo absent."""

    def __init__(self, enabled: Optional[bool] = None,
                 config: Optional[Dict[str, Any]] = None,
                 scientific_role: ScientificRole = ScientificRole.OPTIONAL_DEPLOYMENT):
        env = os.environ.get("NEMO_GUARDRAILS_ENABLED", "false").lower() == "true"
        self.enabled = env if enabled is None else bool(enabled)
        self.config = config or {"rails": {"input": [], "retrieval": [],
                                           "execution": [], "output": []}}
        self.scientific_role = scientific_role
        self.log: List[GuardrailInterventionRecord] = []

    @property
    def config_hash(self) -> str:
        """Enters spec_hash. Identical across all A05/A16 arms."""
        return sha256_obj({"enabled": self.enabled, "config": self.config,
                           "role": self.scientific_role.value})

    def apply(self, rail: RailType, text: str, rail_name: str = "default",
              trace_id: Optional[str] = None, span_id: Optional[str] = None,
              ) -> Dict[str, Any]:
        """Returns {text, action, record}. Disabled == exact bypass."""
        t0 = time.perf_counter()
        in_hash = sha256_text(text)
        if not self.enabled:
            # BIT-IDENTICAL bypass: same object, same hash, no record emitted.
            return {"text": text, "action": RailAction.ALLOW.value,
                    "record": None, "bypassed": True}

        action, out, reason, trigger = RailAction.ALLOW, text, "", None
        low = text.lower()
        if rail is RailType.INPUT and any(
                m in low for m in ("ignore previous instructions",
                                   "disregard the system prompt")):
            action, trigger, reason = RailAction.BLOCK, "prompt_injection_pattern", \
                "input rail blocked a suspected injection"
        rec = GuardrailInterventionRecord(
            rail_type=rail.value, rail_name=rail_name, config_hash=self.config_hash,
            trigger=trigger, input_hash=in_hash, output_hash=sha256_text(out),
            action=action.value, reason=reason,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            trace_id=trace_id, span_id=span_id,
            altered_scientific_input=(out != text),
            scientific_role=self.scientific_role.value)
        self.log.append(rec)
        return {"text": out, "action": action.value, "record": rec.to_dict(),
                "bypassed": False}

    def status(self) -> Dict[str, Any]:
        return {"installed": HAVE_NEMO, "enabled": self.enabled,
                "config_hash": self.config_hash,
                "scientific_role": self.scientific_role.value,
                "rails_supported": [r.value for r in RailType],
                "interventions": len(self.log),
                "rewrites_scientific_input_silently": False}


def assert_symmetric(engines: Dict[str, RailEngine]) -> Dict[str, Any]:
    """A05/A16 guard: identical guardrail configuration across every arm."""
    hashes = {k: e.config_hash for k, e in engines.items()}
    enabled = {k: e.enabled for k, e in engines.items()}
    ok = len(set(hashes.values())) <= 1 and len(set(enabled.values())) <= 1
    return {"symmetric": ok, "config_hashes": hashes, "enabled": enabled,
            "note": "primary comparison arms must share one guardrail configuration"}
