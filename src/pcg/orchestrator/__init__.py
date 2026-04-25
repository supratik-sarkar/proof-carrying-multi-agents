"""
Multi-agent orchestration via LangGraph.

The orchestrator is responsible for:
    1. Wiring the four agents (Prover/Verifier/Attacker/Debugger) into a graph
    2. Routing decisions BASED ON CERTIFICATE STATE, not on hardcoded role labels
       (this is the answer to ICML W2 — "rigid agent roles")
    3. Maintaining the AgenticRuntimeGraph G_t shared by all agents
    4. Registering replay handlers so the Verifier can reproduce all ops

The flow is dynamic: an example only gets a Debugger pass if the Verifier
flags a failure; an Attacker pass only runs in adversarial sweeps. This is
not "Prover -> Verifier -> Attacker -> Debugger every time" — that would be
exactly the rigid pipeline we're criticized for.
"""
from __future__ import annotations

from pcg.orchestrator.langgraph_flow import (
    PCGState,
    build_pcg_graph,
    run_one_example,
)
from pcg.orchestrator.replay_handlers import build_replayer_with_handlers

__all__ = [
    "PCGState",
    "build_pcg_graph",
    "build_replayer_with_handlers",
    "run_one_example",
]
