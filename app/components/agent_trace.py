"""
Streaming agent-trace viewer.

Renders the Prover → Verifier → Checker → Auditor pipeline as a vertical
timeline, with each step's status (pending / running / verified / failed)
animated as the run progresses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import streamlit as st


StepStatus = Literal["pending", "running", "ok", "fail"]


@dataclass
class TraceStep:
    agent: str
    role: str            # short description, e.g. "retrieve evidence"
    status: StepStatus = "pending"
    detail: str = ""     # filled in when status flips to ok/fail
    duration_ms: int = 0


@dataclass
class AgentTrace:
    """Mutable trace that pages mutate while the agents run."""
    steps: list[TraceStep] = field(default_factory=list)

    def add(self, step: TraceStep) -> int:
        self.steps.append(step)
        return len(self.steps) - 1

    def update(
        self, idx: int, *,
        status: StepStatus | None = None,
        detail: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        s = self.steps[idx]
        if status is not None:
            s.status = status
        if detail is not None:
            s.detail = detail
        if duration_ms is not None:
            s.duration_ms = duration_ms


# ---------------------------------------------------------------------------
# Renderer (writes into a Streamlit `st.empty()` placeholder so the page
# can be re-rendered as steps progress)
# ---------------------------------------------------------------------------

_ICONS: dict[StepStatus, str] = {
    "pending": "⚪",
    "running": "🟡",
    "ok": "🟢",
    "fail": "🔴",
}


def render_trace(placeholder: st.delta_generator.DeltaGenerator,
                 trace: AgentTrace) -> None:
    """Re-render the entire trace into the given placeholder."""
    parts: list[str] = ["<div class='pcg-card'>"]
    parts.append(
        "<div style='font-weight:600; margin-bottom:8px;'>"
        "Agent trace</div>"
    )
    if not trace.steps:
        parts.append(
            "<div style='color:var(--pcg-ink-light);'>"
            "Waiting for the first step…</div>"
        )
    else:
        for i, step in enumerate(trace.steps):
            icon = _ICONS[step.status]
            duration = (
                f" · {step.duration_ms} ms"
                if step.status in ("ok", "fail") and step.duration_ms
                else ""
            )
            detail_html = (
                f"<div class='pcg-mono' style='color:var(--pcg-ink-light); "
                f"margin-left:24px; font-size:0.85em;'>{step.detail}</div>"
                if step.detail else ""
            )
            parts.append(
                f"<div style='margin: 6px 0;'>"
                f"{icon} <strong>{step.agent}</strong> · "
                f"{step.role}<span style='color:var(--pcg-ink-light);'>"
                f"{duration}</span>"
                f"{detail_html}"
                f"</div>"
            )
    parts.append("</div>")
    placeholder.markdown("".join(parts), unsafe_allow_html=True)


def make_default_pipeline() -> AgentTrace:
    """The standard PCG-MAS pipeline as TraceSteps (all pending initially)."""
    return AgentTrace(steps=[
        TraceStep("Retriever", "fetch top-k evidence"),
        TraceStep("Prover (Agent 1)", "draft claim with citations"),
        TraceStep("Prover (Agent 2)", "redundant draft with citations"),
        TraceStep("Verifier", "deterministic checker on (Π, Γ)"),
        TraceStep("Auditor", "decompose audit channels"),
        TraceStep("Sealer", "compute signatures S, finalize Z"),
    ])
