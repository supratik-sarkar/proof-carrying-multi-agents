"""
LangGraph-based dynamic orchestrator for PCG-MAS.

The orchestrator answers ICML W2 directly:

    "The proposed multi-agent architecture relies on a fairly hard-coded
     decomposition into roles ... a fixed certificate pipeline."

Our response: roles are *capabilities* (build/check/perturb/intervene), and
the routing edges are conditional on certificate state. An example flows
through the graph based on what its certificate looks like, not based on
its position in a fixed pipeline:

         [Prover] -> [Verifier]
              ^         |
              |         |  if Check failed AND we're in adversarial mode
              |         v
              |    [Debugger] -> (Resp) -> (Risk policy) -> END
              |         |
              |         |  if Resp suggests retrieval/tool problem
              |         v
              +---- [retry with different tool/retriever]

         [Attacker] is OPTIONAL — only invoked under adversarial config

This means a clean example with a passing certificate goes Prover -> Verifier
-> END (3 nodes). An adversarially-tampered certificate goes Prover -> Attacker
-> Verifier (fails) -> Debugger -> END. A failed-clean example may loop back
through Prover with a different retriever. Same code, different paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pcg.backends.base import LLMBackend
from pcg.certificate import GroundingCertificate
from pcg.checker import CheckResult, Checker
from pcg.datasets.base import QAExample
from pcg.eval.meter import Meter
from pcg.graph import AgenticRuntimeGraph


# ---------------------------------------------------------------------------
# State carried through the graph
# ---------------------------------------------------------------------------


@dataclass
class PCGState:
    """The state object that flows through the LangGraph nodes.

    LangGraph will deep-copy this between nodes (or pass by reference depending
    on config). We keep it as a plain dataclass for portability — if LangGraph
    isn't installed, we provide a minimal in-process executor below that uses
    the same state.

    Fields:
        example:        the input QA example
        graph:          the runtime graph G_t shared across this example
        certificate:    the certificate currently under construction/check
        check_result:   most recent verifier output
        retries:        number of Prover retries (capped to prevent loops)
        responsibility: per-component Resp_hat after Debugger pass, if any
        chosen_action:  final action from the risk policy ({answer,verify,...})
        meter:          per-example overhead meter (R5)
        meta:           free-form notes (failure reasons, attack params, ...)
    """

    example: QAExample
    graph: AgenticRuntimeGraph = field(default_factory=AgenticRuntimeGraph)
    certificate: GroundingCertificate | None = None
    check_result: CheckResult | None = None
    retries: int = 0
    responsibility: dict[str, float] = field(default_factory=dict)
    chosen_action: str | None = None
    meter: Meter = field(default_factory=Meter)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    """Configuration for one orchestrator run."""

    max_retries: int = 2
    enable_attacker: bool = False    # turned on only in R1 adversarial sweeps
    enable_debugger: bool = True     # turned on for R3, R4
    attack_kind: Literal["evidence_swap", "schema_break", "policy_violation"] = "evidence_swap"
    # Risk policy parameters (used by the debugger node when triggered)
    risk_lambda: float = 1.0
    h_fa: float = 1.0
    h_ref: float = 0.0


# ---------------------------------------------------------------------------
# Node functions (decoupled from LangGraph so they're testable in isolation)
# ---------------------------------------------------------------------------


def prover_node(
    state: PCGState,
    *,
    prover_fn: Callable[[PCGState], PCGState],
) -> PCGState:
    """Prover builds the certificate. The actual logic lives in pcg.agents.prover."""
    with state.meter.phase("prover"):
        return prover_fn(state)


def attacker_node(
    state: PCGState,
    *,
    attacker_fn: Callable[[PCGState], PCGState],
) -> PCGState:
    """Optional adversarial perturbation. Only invoked if config.enable_attacker."""
    with state.meter.phase("attacker"):
        return attacker_fn(state)


def verifier_node(
    state: PCGState,
    *,
    checker: Checker,
) -> PCGState:
    """Run Check(Z; G_t) and store the structured result.

    If the check fails, increment `state.retries` HERE (inside a node).
    LangGraph persists state mutations made by node functions; mutations
    inside conditional-edge functions are silently discarded. Centralizing
    the increment here is the only way the retry counter can grow.
    """
    with state.meter.phase("verifier"):
        if state.certificate is None:
            return state
        state.check_result = checker.check(state.certificate, state.graph)
        if not state.check_result.passed:
            state.retries += 1
    return state


def debugger_node(
    state: PCGState,
    *,
    debugger_fn: Callable[[PCGState], PCGState],
) -> PCGState:
    """Run mask-and-replay interventions and select an action via risk policy."""
    with state.meter.phase("debugger"):
        return debugger_fn(state)


# ---------------------------------------------------------------------------
# Routing functions — these are what make the flow dynamic
# ---------------------------------------------------------------------------


def _post_prover_route(state: PCGState, cfg: OrchestratorConfig) -> str:
    """After Prover, optionally invoke Attacker, otherwise go to Verifier."""
    if cfg.enable_attacker:
        return "attacker"
    return "verifier"


def _post_verifier_route(state: PCGState, cfg: OrchestratorConfig) -> str:
    """After Verifier:
        - if Check passed and not in debugger-mode -> END
        - if Check passed and debugger-mode is on -> Debugger (for R3/R4)
        - if Check failed and we have retries left -> back to Prover
        - if Check failed and no retries left -> Debugger (for diagnosis)

    Note: by the time we get here, verifier_node has ALREADY incremented
    state.retries on failure. So state.retries is "number of failed checks
    so far" — we permit going back to prover as long as that count does not
    exceed cfg.max_retries.
    """
    if state.check_result is None:
        return "end"
    passed = state.check_result.passed
    if passed:
        return "debugger" if cfg.enable_debugger else "end"
    # state.retries was already incremented in verifier_node; allow up to
    # max_retries failed-and-retried attempts.
    if state.retries <= cfg.max_retries:
        return "prover"     # retry with a different retriever / branch
    return "debugger" if cfg.enable_debugger else "end"


# ---------------------------------------------------------------------------
# Graph builder — uses LangGraph if installed, else in-process executor
# ---------------------------------------------------------------------------


def build_pcg_graph(
    cfg: OrchestratorConfig,
    *,
    prover_fn: Callable[[PCGState], PCGState],
    attacker_fn: Callable[[PCGState], PCGState] | None = None,
    debugger_fn: Callable[[PCGState], PCGState] | None = None,
    checker: Checker,
) -> Callable[[PCGState], PCGState]:
    """Build a callable that runs one example through the orchestrator.

    Tries LangGraph first; falls back to a simple in-process executor that
    follows the same routing edges. The fallback exists so that smoke tests
    don't require LangGraph to be installed (it's a heavy import).
    """
    try:
        from langgraph.graph import END, StateGraph
        return _build_with_langgraph(
            cfg, prover_fn=prover_fn, attacker_fn=attacker_fn,
            debugger_fn=debugger_fn, checker=checker,
            END=END, StateGraph=StateGraph,
        )
    except ImportError:
        return _build_in_process(
            cfg, prover_fn=prover_fn, attacker_fn=attacker_fn,
            debugger_fn=debugger_fn, checker=checker,
        )


def _build_with_langgraph(
    cfg: OrchestratorConfig,
    *,
    prover_fn,
    attacker_fn,
    debugger_fn,
    checker,
    END,
    StateGraph,
) -> Callable[[PCGState], PCGState]:
    """LangGraph-backed orchestrator. Each node mutates the PCGState and the
    routing functions return the next node name.
    """
    sg: Any = StateGraph(PCGState)
    sg.add_node("prover", lambda s: prover_node(s, prover_fn=prover_fn))
    if attacker_fn is not None:
        sg.add_node("attacker", lambda s: attacker_node(s, attacker_fn=attacker_fn))
    sg.add_node("verifier", lambda s: verifier_node(s, checker=checker))
    if debugger_fn is not None:
        sg.add_node("debugger", lambda s: debugger_node(s, debugger_fn=debugger_fn))

    sg.set_entry_point("prover")

    # Conditional edges
    def route_after_prover(s: PCGState) -> str:
        nxt = _post_prover_route(s, cfg)
        return nxt if nxt in {"attacker", "verifier"} and (
            attacker_fn is not None or nxt != "attacker"
        ) else "verifier"

    sg.add_conditional_edges("prover", route_after_prover, {
        "attacker": "attacker" if attacker_fn is not None else "verifier",
        "verifier": "verifier",
    })

    if attacker_fn is not None:
        sg.add_edge("attacker", "verifier")

    def route_after_verifier(s: PCGState) -> str:
        # IMPORTANT: do NOT mutate state in this routing function — LangGraph
        # discards mutations made inside conditional-edge functions. The
        # retry counter is incremented inside verifier_node (where it works).
        nxt = _post_verifier_route(s, cfg)
        if nxt == "prover":
            return "prover"
        if nxt == "debugger" and debugger_fn is not None:
            return "debugger"
        return "end"

    sg.add_conditional_edges("verifier", route_after_verifier, {
        "prover": "prover",
        "debugger": "debugger" if debugger_fn is not None else END,
        "end": END,
    })

    if debugger_fn is not None:
        sg.add_edge("debugger", END)

    compiled = sg.compile()

    def run(state: PCGState) -> PCGState:
        result = compiled.invoke(state)
        # LangGraph returns a dict-like; coerce back to PCGState if needed
        if isinstance(result, dict):
            return PCGState(**result) if "example" in result else state
        return result

    return run


def _build_in_process(
    cfg: OrchestratorConfig,
    *,
    prover_fn,
    attacker_fn,
    debugger_fn,
    checker,
) -> Callable[[PCGState], PCGState]:
    """Minimal in-process executor with identical routing semantics. Used
    when LangGraph isn't installed (e.g., during the smoke test).
    """

    def run(state: PCGState) -> PCGState:
        # Hard cap on iterations — prevents pathological loops in retry logic.
        for _step in range(8):
            state = prover_node(state, prover_fn=prover_fn)
            if cfg.enable_attacker and attacker_fn is not None:
                state = attacker_node(state, attacker_fn=attacker_fn)
            state = verifier_node(state, checker=checker)
            # verifier_node already incremented state.retries on failure; we
            # only need to read the routing decision here.
            nxt = _post_verifier_route(state, cfg)
            if nxt == "prover":
                continue
            if nxt == "debugger" and debugger_fn is not None:
                state = debugger_node(state, debugger_fn=debugger_fn)
            break
        return state

    return run


# ---------------------------------------------------------------------------
# High-level convenience: assemble all four agents and run one example
# ---------------------------------------------------------------------------


def run_one_example(
    example: QAExample,
    *,
    backend: LLMBackend,
    checker: Checker,
    cfg: OrchestratorConfig | None = None,
    prover_fn: Callable[[PCGState], PCGState] | None = None,
    attacker_fn: Callable[[PCGState], PCGState] | None = None,
    debugger_fn: Callable[[PCGState], PCGState] | None = None,
) -> PCGState:
    """One-shot entry point used by experiment scripts.

    If `prover_fn` / `attacker_fn` / `debugger_fn` are not supplied, sensible
    defaults are imported from `pcg.agents`. The default Prover uses BM25 +
    the supplied backend; the default Attacker does evidence-swap; the default
    Debugger runs single-component Resp interventions.
    """
    cfg = cfg or OrchestratorConfig()
    if prover_fn is None:
        from pcg.agents.prover import build_default_prover
        prover_fn = build_default_prover(backend=backend)
    if attacker_fn is None and cfg.enable_attacker:
        from pcg.agents.attacker import build_default_attacker
        attacker_fn = build_default_attacker(kind=cfg.attack_kind)
    if debugger_fn is None and cfg.enable_debugger:
        from pcg.agents.debugger import build_default_debugger
        debugger_fn = build_default_debugger(checker=checker, cfg=cfg)

    state = PCGState(example=example)
    runner = build_pcg_graph(
        cfg, prover_fn=prover_fn, attacker_fn=attacker_fn,
        debugger_fn=debugger_fn, checker=checker,
    )
    return runner(state)
