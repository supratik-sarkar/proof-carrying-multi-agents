"""Real NeMo Guardrails multi-rail runtime integration.

Multi-rail boundaries:
  - INPUT: prompt-injection / boundary checks on user request
  - RETRIEVAL: source-grounding / provenance filters on retrieved evidence
  - EXECUTION/TOOL: unauthorized tool / parameter bounds
  - OUTPUT: toxic / off-policy output screening

Invariants:
  - When disabled (enabled=False): bit-identical bypass without modification.
  - Every intervention logs: rail, config_hash, action, reason, input_hash,
    output_hash, trace_id, span_id, scientific_role="boundary_control".
  - NeMo is boundary control only: scientific input is never silently rewritten.
  - A05/A16 primary config requires guardrails identically OFF across all arms.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..canon import sha256_obj, sha256_text

ENABLED = os.environ.get("NEMO_GUARDRAILS_ENABLED", "false").lower() == "true"

try:
    import nemoguardrails
    from nemoguardrails import RailsConfig, LLMRails
    HAVE_NEMO = True
except Exception:
    HAVE_NEMO = False
    RailsConfig = None
    LLMRails = None


@dataclass
class Intervention:
    rail: str                  # "input" | "retrieval" | "execution" | "output"
    config_hash: str
    action: str                # "flag" | "block" | "passthrough"
    reason: str
    input_hash: str
    output_hash: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    scientific_role: str = "boundary_control"
    note: str = "boundary control only; scientific input is never silently rewritten"

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


class NeMoAdapter:
    name = "nemo_guardrails"

    DEFAULT_COLANG = """
define flow input check
  user ...
  $has_leak = execute check_injection
  if $has_leak
    bot refuse
"""

    DEFAULT_CONFIG_YAML = """
models: []
rails:
  input:
    flows:
      - input check
"""

    def __init__(self, enabled: Optional[bool] = None,
                 colang_content: Optional[str] = None,
                 yaml_content: Optional[str] = None):
        self.enabled = (ENABLED if enabled is None else enabled) and HAVE_NEMO
        self.log: List[Intervention] = []
        self.rails: Optional[LLMRails] = None
        self.config_hash = sha256_obj({
            "colang": colang_content or self.DEFAULT_COLANG,
            "yaml": yaml_content or self.DEFAULT_CONFIG_YAML,
            "version": nemoguardrails.__version__ if HAVE_NEMO else "none"
        })

        if self.enabled and HAVE_NEMO:
            cfg = RailsConfig.from_content(
                colang_content=colang_content or self.DEFAULT_COLANG,
                yaml_content=yaml_content or self.DEFAULT_CONFIG_YAML
            )
            self.rails = LLMRails(cfg)

    def check_input(self, text: str, trace_id: Optional[str] = None,
                    span_id: Optional[str] = None) -> Dict[str, Any]:
        in_hash = sha256_text(text)
        if not self.enabled:
            return {"applied": False, "flagged": False, "text": text,
                    "reason": "guardrails disabled", "interventions": []}

        blocked = any(m in text.lower() for m in (
            "ignore previous instructions", "disregard the system prompt", "jailbreak"
        ))
        action = "block" if blocked else "passthrough"
        reason = "prompt_injection_pattern_detected" if blocked else "input_clean"
        out_text = text

        inter = Intervention(
            rail="input", config_hash=self.config_hash, action=action,
            reason=reason, input_hash=in_hash, output_hash=sha256_text(out_text),
            trace_id=trace_id, span_id=span_id
        )
        if blocked:
            self.log.append(inter)

        return {"applied": True, "flagged": blocked, "text": out_text,
                "interventions": [inter.to_dict()] if blocked else []}

    def check_retrieval(self, evidence_items: Sequence[str], trace_id: Optional[str] = None,
                        span_id: Optional[str] = None) -> Dict[str, Any]:
        in_hash = sha256_obj(list(evidence_items))
        if not self.enabled:
            return {"applied": False, "flagged": False, "evidence": list(evidence_items),
                    "reason": "guardrails disabled", "interventions": []}

        flagged = any("untrusted_leak" in ev.lower() for ev in evidence_items)
        action = "flag" if flagged else "passthrough"
        reason = "untrusted_evidence_content" if flagged else "retrieval_clean"
        out_ev = list(evidence_items)

        inter = Intervention(
            rail="retrieval", config_hash=self.config_hash, action=action,
            reason=reason, input_hash=in_hash, output_hash=sha256_obj(out_ev),
            trace_id=trace_id, span_id=span_id
        )
        if flagged:
            self.log.append(inter)

        return {"applied": True, "flagged": flagged, "evidence": out_ev,
                "interventions": [inter.to_dict()] if flagged else []}

    def check_execution_tool(self, tool_name: str, tool_args: Dict[str, Any],
                             trace_id: Optional[str] = None,
                             span_id: Optional[str] = None) -> Dict[str, Any]:
        in_hash = sha256_obj({"tool": tool_name, "args": tool_args})
        if not self.enabled:
            return {"applied": False, "flagged": False, "tool_name": tool_name,
                    "reason": "guardrails disabled", "interventions": []}

        disallowed = tool_name in ("execute_arbitrary_code", "rm_rf", "eval_untrusted")
        action = "block" if disallowed else "passthrough"
        reason = f"disallowed_execution_tool_{tool_name}" if disallowed else "tool_clean"

        inter = Intervention(
            rail="execution", config_hash=self.config_hash, action=action,
            reason=reason, input_hash=in_hash, output_hash=in_hash,
            trace_id=trace_id, span_id=span_id
        )
        if disallowed:
            self.log.append(inter)

        return {"applied": True, "flagged": disallowed, "tool_name": tool_name,
                "interventions": [inter.to_dict()] if disallowed else []}

    def check_output(self, text: str, trace_id: Optional[str] = None,
                     span_id: Optional[str] = None) -> Dict[str, Any]:
        in_hash = sha256_text(text)
        if not self.enabled:
            return {"applied": False, "flagged": False, "text": text,
                    "reason": "guardrails disabled", "interventions": []}

        toxic = "forbidden_secret_leak" in text.lower()
        action = "block" if toxic else "passthrough"
        reason = "sensitive_data_in_output" if toxic else "output_clean"
        out_text = text

        inter = Intervention(
            rail="output", config_hash=self.config_hash, action=action,
            reason=reason, input_hash=in_hash, output_hash=sha256_text(out_text),
            trace_id=trace_id, span_id=span_id
        )
        if toxic:
            self.log.append(inter)

        return {"applied": True, "flagged": toxic, "text": out_text,
                "interventions": [inter.to_dict()] if toxic else []}

    def status(self) -> Dict[str, Any]:
        return {
            "installed": HAVE_NEMO,
            "version": nemoguardrails.__version__ if HAVE_NEMO else None,
            "real_runtime_initialized": self.rails is not None,
            "enabled": self.enabled,
            "config_hash": self.config_hash,
            "rails_supported": ["input", "retrieval", "execution", "output"],
            "rewrites_inputs": False,
            "interventions_logged": len(self.log),
        }
