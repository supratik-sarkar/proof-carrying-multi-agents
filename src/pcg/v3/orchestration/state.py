"""Typed execution state for the PCG-MAS graph."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

NODES = ["request", "task_normalization", "generation", "evidence_acquisition",
         "evidence_commitment", "execution_contract", "entailment_check",
         "replay_check", "audit", "dependence", "controller", "terminal"]
TERMINALS = ["accepted", "repaired", "escalated", "refused"]


@dataclass
class GraphState:
    run_id: str
    request: str
    node: str = "request"
    terminal: Optional[str] = None
    claim: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    evidence_hashes: List[str] = field(default_factory=list)
    v_h: Optional[bool] = None
    v_pi: Optional[bool] = None
    v_gamma: Optional[bool] = None
    v_entail: Optional[bool] = None
    check: Optional[bool] = None
    channels: Dict[str, bool] = field(default_factory=dict)
    policy_decision: Optional[Dict[str, Any]] = None
    dependence: Optional[Dict[str, Any]] = None
    controller_action: Optional[str] = None
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=lambda: {
        "latency_ms": 0.0, "tokens_in": 0, "tokens_out": 0, "model_calls": 0,
        "retrieval_calls": 0, "tool_calls": 0, "checker_calls": 0, "replay_calls": 0})
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
