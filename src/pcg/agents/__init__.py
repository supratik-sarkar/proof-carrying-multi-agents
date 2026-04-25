"""
The four PCG-MAS agents (Appendix D.4 of the paper).

    - Prover    : builds the unified certificate Z = (c, S, Pi, Gamma, p, meta)
    - Verifier  : delegated to pcg.checker.Checker (no agent class needed)
    - Attacker  : tampers with logs / evidence / schemas to test soundness
    - Debugger  : runs do-interventions for diagnosis + risk-policy choice

Each agent is a pure function PCGState -> PCGState (or set thereof) so that
they compose cleanly through the orchestrator.
"""
from __future__ import annotations

from pcg.agents.attacker import build_default_attacker
from pcg.agents.debugger import build_default_debugger
from pcg.agents.prover import build_default_prover

__all__ = [
    "build_default_attacker",
    "build_default_debugger",
    "build_default_prover",
]
