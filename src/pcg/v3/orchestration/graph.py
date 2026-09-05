"""Explicit PCG-MAS execution graph.

LangGraph is used as the typed orchestration layer WHEN INSTALLED; the identical
deterministic sequence runs without it. There are no hidden autonomous loops, no
unbounded retries, every transition is explicit, and every node emits provenance.
The manuscript semantics -- not LangGraph -- define PCG-MAS.
"""
from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..canon import sha256_text
from ..channels import check as check_predicate
from ..policy.local import LocalPolicyBackend
from ..providers.registry import get_provider
from ..telemetry.otel import span
from .state import GraphState, NODES, TERMINALS

try:                                   # optional
    from langgraph.graph import END, StateGraph  # type: ignore
    HAVE_LANGGRAPH = True
except Exception:
    HAVE_LANGGRAPH = False


def _prov(st: GraphState, node: str, **fields):
    st.provenance.append({"node": node, "ts_ms": round(time.time() * 1000, 1), **fields})


# --------------------------------------------------------------------- nodes
def n_task_normalization(st: GraphState) -> GraphState:
    with span("task_normalization"):
        st.claim = st.request.strip()
        st.node = "task_normalization"
        _prov(st, st.node, claim_hash=sha256_text(st.claim or "")[:16])
    return st


def n_generation(st: GraphState, provider=None) -> GraphState:
    with span("generation"):
        p = provider or get_provider("offline_mock")
        r = p.generate(st.claim or "")
        st.cost["model_calls"] += 1
        st.cost["tokens_in"] += r.tokens_in
        st.cost["tokens_out"] += r.tokens_out
        st.cost["latency_ms"] += r.latency_ms
        st.node = "generation"
        _prov(st, st.node, backend_fingerprint=r.backend_fingerprint,
              raw_output_sha256=r.raw_output_sha256)
    return st


def n_evidence_acquisition(st: GraphState, evidence: Optional[List[str]] = None) -> GraphState:
    with span("evidence_acquisition"):
        ev = evidence or [f"ev::{sha256_text(st.claim or '')[:8]}"]
        st.evidence_ids = ev
        st.cost["retrieval_calls"] += 1
        st.node = "evidence_acquisition"
        _prov(st, st.node, n_evidence=len(ev))
    return st


def n_evidence_commitment(st: GraphState) -> GraphState:
    with span("evidence_commitment"):
        st.evidence_hashes = [sha256_text(e) for e in st.evidence_ids]
        st.v_h = len(st.evidence_hashes) == len(st.evidence_ids) and all(st.evidence_hashes)
        st.channels["int_fail"] = not st.v_h
        st.node = "evidence_commitment"
        _prov(st, st.node, v_h=st.v_h)
    return st


def n_execution_contract(st: GraphState, backend=None, request: Optional[Dict] = None) -> GraphState:
    with span("execution_contract"):
        b = backend or LocalPolicyBackend()
        d = b.evaluate(request or {"actor": "prover", "action": "answer"})
        st.policy_decision = d.to_dict()
        st.v_gamma = d.allowed
        st.channels["cov_gap"] = not d.allowed and d.rule_matched == "schema_validation"
        st.node = "execution_contract"
        _prov(st, st.node, v_gamma=st.v_gamma, bundle_sha256=d.bundle_sha256)
    return st


def n_entailment_check(st: GraphState, verdict: Optional[bool] = None) -> GraphState:
    with span("entailment_check"):
        st.v_entail = True if verdict is None else bool(verdict)
        st.cost["checker_calls"] += 1
        st.channels["check_fail"] = st.channels.get("check_fail", False)
        st.node = "entailment_check"
        _prov(st, st.node, v_entail=st.v_entail,
              note="checker-relative: reproducible under version pinning, not objectively true")
    return st


def n_replay_check(st: GraphState, replay_ok: Optional[bool] = None,
                   drift: Optional[bool] = None) -> GraphState:
    with span("replay_check"):
        st.v_pi = True if replay_ok is None else bool(replay_ok)
        st.cost["replay_calls"] += 1
        st.channels["replay_fail"] = not st.v_pi
        st.channels["drift_fail"] = bool(drift)
        st.node = "replay_check"
        _prov(st, st.node, v_pi=st.v_pi, drift=bool(drift))
    return st


def n_audit(st: GraphState) -> GraphState:
    with span("audit"):
        st.check = check_predicate(st.v_h, st.v_pi, st.v_gamma, st.v_entail)
        st.node = "audit"
        _prov(st, st.node, check=st.check,
              n_channels_fired=sum(1 for v in st.channels.values() if v))
    return st


def n_dependence(st: GraphState, dependence: Optional[Dict[str, Any]] = None) -> GraphState:
    with span("dependence"):
        st.dependence = dependence or {"state": "INSUFFICIENT_EVIDENCE",
                                       "note": "single-branch run; no extrapolation credited"}
        st.node = "dependence"
        _prov(st, st.node, gate=st.dependence.get("state"))
    return st


def n_controller(st: GraphState, risk: float = 0.0, model=None) -> GraphState:
    with span("controller"):
        from ..science.controller import CostModel
        m = model or CostModel()
        st.controller_action = m.policy(risk) if st.check else "Refuse"
        st.node = "controller"
        _prov(st, st.node, action=st.controller_action, risk=risk)
    return st


def n_terminal(st: GraphState) -> GraphState:
    with span("terminal"):
        st.terminal = {"Answer": "accepted", "Verify": "escalated",
                       "Escalate": "escalated", "Refuse": "refused"}.get(
                           st.controller_action or "Refuse", "refused")
        st.node = "terminal"
        _prov(st, st.node, terminal=st.terminal)
    return st


SEQUENCE: List[Tuple[str, Callable]] = [
    ("task_normalization", n_task_normalization),
    ("generation", n_generation),
    ("evidence_acquisition", n_evidence_acquisition),
    ("evidence_commitment", n_evidence_commitment),
    ("execution_contract", n_execution_contract),
    ("entailment_check", n_entailment_check),
    ("replay_check", n_replay_check),
    ("audit", n_audit),
    ("dependence", n_dependence),
    ("controller", n_controller),
    ("terminal", n_terminal),
]


def run_graph(request: str, run_id: str = "run", **node_kwargs) -> GraphState:
    """Deterministic execution. Identical with or without LangGraph installed."""
    st = GraphState(run_id=run_id, request=request)
    for name, fn in SEQUENCE:
        kw = node_kwargs.get(name, {})
        st = fn(st, **kw)
    return st


def build_langgraph():
    """Return a compiled LangGraph with the same explicit transitions, or None."""
    if not HAVE_LANGGRAPH:
        return None
    g = StateGraph(dict)
    prev = None
    for name, fn in SEQUENCE:
        g.add_node(name, lambda s, _fn=fn: _fn(GraphState(**s)).to_dict())
        if prev is None:
            g.set_entry_point(name)
        else:
            g.add_edge(prev, name)
        prev = name
    g.add_edge(prev, END)
    return g.compile()


def status() -> Dict[str, Any]:
    return {"langgraph_installed": HAVE_LANGGRAPH,
            "nodes": [n for n, _ in SEQUENCE], "terminals": TERMINALS,
            "core_runs_without_langgraph": True}
