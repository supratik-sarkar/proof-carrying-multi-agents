"""PCG-MAS conditional execution state machine.

A genuine typed conditional topology, not a sequential wrapper. Every transition
is explicit, every loop is hard-bounded, and there is no autonomous agent loop.
LangGraph drives it when installed; the identical deterministic machine runs
without it, because the manuscript semantics -- not LangGraph -- define PCG-MAS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..canon import sha256_obj
from ..channels import check as check_predicate
from ..exec.policy_status import PolicyEvalStatus, from_backend
from ..exec.retry import MAX_ATTEMPTS
from ..telemetry.otel import span

try:
    from langgraph.graph import END, StateGraph  # type: ignore
    HAVE_LANGGRAPH = True
except Exception:
    HAVE_LANGGRAPH = False


class Node(str, Enum):
    TASK_NORMALIZATION = "task_normalization"
    INPUT_GUARDRAIL = "input_guardrail"
    RETRIEVAL = "retrieval"
    RETRIEVAL_GUARDRAIL = "retrieval_guardrail"
    GENERATION = "generation"
    EVIDENCE_CANONICALIZATION = "evidence_canonicalization"
    EVIDENCE_COMMITMENT = "evidence_commitment"
    EXECUTION_POLICY_CHECK = "execution_policy_check"
    ENTAILMENT_CHECK = "entailment_check"
    REPLAY_CHECK = "replay_check"
    ACCEPTANCE_COMPOSITION = "acceptance_composition"
    AUDIT = "audit"
    DEPENDENCE_ANALYSIS = "dependence_analysis"
    CONTROLLER = "controller"
    REPAIR_OR_ESCALATION = "repair_or_escalation"
    OUTPUT_GUARDRAIL = "output_guardrail"
    ARTIFACT_PERSISTENCE = "artifact_persistence"
    TERMINAL = "terminal"


class Terminal(str, Enum):
    ACCEPTED = "accepted"
    REPAIRED = "repaired"
    ESCALATED = "escalated"
    REFUSED = "refused"


#: Hard bounds. There is no unbounded retry and no autonomous loop.
BOUNDS = {"generation_attempts": MAX_ATTEMPTS, "repair_rounds": 1}


@dataclass
class GraphState:
    run_id: str
    request: str
    node: str = Node.TASK_NORMALIZATION.value
    terminal: Optional[str] = None
    # conjuncts
    v_h: Optional[bool] = None
    v_pi: Optional[bool] = None
    v_gamma: Optional[bool] = None
    v_entail: Optional[bool] = None
    check: Optional[bool] = None
    policy_eval_status: Optional[str] = None
    # channels
    channels: Dict[str, bool] = field(default_factory=dict)
    # bounded counters
    generation_attempts: int = 0
    repair_rounds: int = 0
    # artifacts
    claim: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    evidence_hashes: List[str] = field(default_factory=list)
    certificate_root: Optional[str] = None
    dependence: Optional[Dict[str, Any]] = None
    controller_action: Optional[str] = None
    guardrails: List[Dict[str, Any]] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    visited: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items()}


# ------------------------------------------------------------------ routing
def route(state: GraphState) -> str:
    """Explicit conditional edges. Returns the next node, or TERMINAL."""
    n = state.node

    if n == Node.TASK_NORMALIZATION.value:
        return Node.INPUT_GUARDRAIL.value
    if n == Node.INPUT_GUARDRAIL.value:
        return Node.RETRIEVAL.value
    if n == Node.RETRIEVAL.value:
        return Node.RETRIEVAL_GUARDRAIL.value
    if n == Node.RETRIEVAL_GUARDRAIL.value:
        return Node.GENERATION.value

    if n == Node.GENERATION.value:
        # generation failure -> bounded retry -> REFUSE on exhaustion
        if state.errors and state.generation_attempts < BOUNDS["generation_attempts"]:
            return Node.GENERATION.value
        if state.errors:
            state.terminal = Terminal.REFUSED.value
            return Node.TERMINAL.value
        return Node.EVIDENCE_CANONICALIZATION.value

    if n == Node.EVIDENCE_CANONICALIZATION.value:
        return Node.EVIDENCE_COMMITMENT.value
    if n == Node.EVIDENCE_COMMITMENT.value:
        return Node.EXECUTION_POLICY_CHECK.value

    if n == Node.EXECUTION_POLICY_CHECK.value:
        # policy violation OR indeterminate -> controller (fails closed either way,
        # but the two are distinguished downstream and never conflated)
        if state.v_gamma is not True:
            return Node.CONTROLLER.value
        return Node.ENTAILMENT_CHECK.value

    if n == Node.ENTAILMENT_CHECK.value:
        if state.v_entail is False and state.repair_rounds < BOUNDS["repair_rounds"]:
            return Node.REPAIR_OR_ESCALATION.value
        return Node.REPLAY_CHECK.value

    if n == Node.REPLAY_CHECK.value:
        return Node.ACCEPTANCE_COMPOSITION.value
    if n == Node.ACCEPTANCE_COMPOSITION.value:
        return Node.AUDIT.value
    if n == Node.AUDIT.value:
        return Node.DEPENDENCE_ANALYSIS.value
    if n == Node.DEPENDENCE_ANALYSIS.value:
        return Node.CONTROLLER.value

    if n == Node.CONTROLLER.value:
        if state.controller_action in ("Verify", "Escalate") and \
           state.repair_rounds < BOUNDS["repair_rounds"]:
            return Node.REPAIR_OR_ESCALATION.value
        return Node.OUTPUT_GUARDRAIL.value

    if n == Node.REPAIR_OR_ESCALATION.value:
        # one bounded repair round, then always forward -- never a loop
        return Node.ENTAILMENT_CHECK.value if state.v_entail is False \
            else Node.OUTPUT_GUARDRAIL.value

    if n == Node.OUTPUT_GUARDRAIL.value:
        return Node.ARTIFACT_PERSISTENCE.value
    if n == Node.ARTIFACT_PERSISTENCE.value:
        return Node.TERMINAL.value
    return Node.TERMINAL.value


def terminal_of(state: GraphState) -> str:
    if state.terminal:
        return state.terminal
    if state.check is True and state.controller_action == "Answer":
        return Terminal.ACCEPTED.value
    if state.repair_rounds > 0 and state.check is True:
        return Terminal.REPAIRED.value
    if state.controller_action in ("Verify", "Escalate"):
        return Terminal.ESCALATED.value
    return Terminal.REFUSED.value


ALL_NODES: Tuple[str, ...] = tuple(n.value for n in Node)
ALL_TERMINALS: Tuple[str, ...] = tuple(t.value for t in Terminal)


def topology() -> Dict[str, Any]:
    """Machine-readable topology for the reviewer UI. No semantics in TypeScript."""
    edges: List[Dict[str, str]] = []
    spec = {
        Node.TASK_NORMALIZATION: [(Node.INPUT_GUARDRAIL, "always")],
        Node.INPUT_GUARDRAIL: [(Node.RETRIEVAL, "always")],
        Node.RETRIEVAL: [(Node.RETRIEVAL_GUARDRAIL, "always")],
        Node.RETRIEVAL_GUARDRAIL: [(Node.GENERATION, "always")],
        Node.GENERATION: [(Node.GENERATION, "error and attempts < 3"),
                          (Node.TERMINAL, "error and attempts exhausted -> REFUSE"),
                          (Node.EVIDENCE_CANONICALIZATION, "ok")],
        Node.EVIDENCE_CANONICALIZATION: [(Node.EVIDENCE_COMMITMENT, "always")],
        Node.EVIDENCE_COMMITMENT: [(Node.EXECUTION_POLICY_CHECK, "always")],
        Node.EXECUTION_POLICY_CHECK: [(Node.CONTROLLER, "V_Gamma is not True"),
                                      (Node.ENTAILMENT_CHECK, "V_Gamma is True")],
        Node.ENTAILMENT_CHECK: [(Node.REPAIR_OR_ESCALATION, "V_vdash False and repair budget"),
                                (Node.REPLAY_CHECK, "otherwise")],
        Node.REPLAY_CHECK: [(Node.ACCEPTANCE_COMPOSITION, "always")],
        Node.ACCEPTANCE_COMPOSITION: [(Node.AUDIT, "always")],
        Node.AUDIT: [(Node.DEPENDENCE_ANALYSIS, "always")],
        Node.DEPENDENCE_ANALYSIS: [(Node.CONTROLLER, "always")],
        Node.CONTROLLER: [(Node.REPAIR_OR_ESCALATION, "Verify/Escalate and repair budget"),
                          (Node.OUTPUT_GUARDRAIL, "otherwise")],
        Node.REPAIR_OR_ESCALATION: [(Node.ENTAILMENT_CHECK, "recheck once"),
                                    (Node.OUTPUT_GUARDRAIL, "forward")],
        Node.OUTPUT_GUARDRAIL: [(Node.ARTIFACT_PERSISTENCE, "always")],
        Node.ARTIFACT_PERSISTENCE: [(Node.TERMINAL, "always")],
    }
    for src, outs in spec.items():
        for dst, cond in outs:
            edges.append({"from": src.value, "to": dst.value, "condition": cond})
    return {"nodes": list(ALL_NODES), "terminals": list(ALL_TERMINALS),
            "edges": edges, "bounds": BOUNDS,
            "langgraph_installed": HAVE_LANGGRAPH,
            "core_runs_without_langgraph": True,
            "topology_hash": sha256_obj({"n": ALL_NODES, "e": edges, "b": BOUNDS})}
