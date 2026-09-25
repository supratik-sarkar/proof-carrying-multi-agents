"""LangSmith as an optional VIEW of the OTel ontology -- never a second ontology.

OTel spans map 1:1 onto LangSmith runs; trace/span ids are carried across so the
two views correlate. When LANGSMITH_TRACING=false there must be ZERO network
traffic, which is asserted at the socket level rather than by trusting that
`flush()` was not called. Reviewer-anonymous mode forces it OFF.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .hierarchy import Span

SENSITIVE_KEYS = ("evidence", "evidence_hashes", "prompt", "raw_output", "claim",
                  "api_key", "authorization", "policy_input")


def tracing_enabled() -> bool:
    if os.environ.get("PCG_ANONYMOUS_REVIEW", "false").lower() == "true":
        return False                      # anonymous mode forces OFF
    return os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"


def upload_evidence_enabled() -> bool:
    return os.environ.get("LANGSMITH_UPLOAD_EVIDENCE", "false").lower() == "true"


def scrub(d: Dict[str, Any]) -> Dict[str, Any]:
    if upload_evidence_enabled():
        return dict(d)
    return {k: ("***WITHHELD***" if any(s in k.lower() for s in SENSITIVE_KEYS) else v)
            for k, v in d.items()}


@dataclass
class LangSmithRun:
    """One OTel span projected into LangSmith's run model."""
    id: str
    trace_id: str
    parent_run_id: Optional[str]
    name: str
    run_type: str
    start_time_ms: float
    end_time_ms: Optional[float]
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


RUN_TYPE = {"graph": "chain", "node": "chain", "operation": "tool",
            "experiment": "chain", "cell": "chain", "seed": "chain",
            "example": "chain", "system": "chain"}
OP_RUN_TYPE = {"provider": "llm", "retrieval": "retriever", "tool": "tool",
               "checker": "tool", "policy": "tool", "replay": "tool",
               "guardrail": "tool"}


def span_to_run(s: Span) -> LangSmithRun:
    rt = RUN_TYPE.get(s.level, "chain")
    kind = s.attributes.get("pcg.op_kind")
    if s.level == "operation" and kind in OP_RUN_TYPE:
        rt = OP_RUN_TYPE[kind]
    return LangSmithRun(
        id=s.span_id, trace_id=s.trace_id, parent_run_id=s.parent_id, name=s.name,
        run_type=rt, start_time_ms=s.start_ms,
        end_time_ms=(s.start_ms + s.duration_ms) if s.duration_ms is not None else None,
        extra=scrub({**s.attributes, "otel_span_id": s.span_id,
                     "otel_trace_id": s.trace_id, "links": s.links}),
        error=s.exception)


class Sink(Protocol):
    def send(self, runs: List[Dict[str, Any]]) -> None: ...


class NullSink:
    """Used for tests: records without touching a socket."""
    def __init__(self) -> None:
        self.received: List[Dict[str, Any]] = []

    def send(self, runs: List[Dict[str, Any]]) -> None:
        self.received.extend(runs)


class LangSmithExporter:
    def __init__(self, project: str = "pcg-mas-v3", sink: Optional[Sink] = None,
                 enabled: Optional[bool] = None):
        self.project = project
        self.enabled = tracing_enabled() if enabled is None else bool(enabled)
        self.sink = sink

    def export(self, spans: List[Span]) -> Dict[str, Any]:
        """No-op when disabled. Opens no socket, constructs no client."""
        if not self.enabled:
            return {"exported": 0, "reason": "LANGSMITH_TRACING=false",
                    "network_calls": 0}
        runs = [span_to_run(s).to_dict() for s in spans]
        if self.sink is not None:
            self.sink.send(runs)
            return {"exported": len(runs), "sink": type(self.sink).__name__,
                    "network_calls": 0}
        # A real client would be constructed lazily here; not in this pass.
        return {"exported": 0, "reason": "no sink configured; no client constructed",
                "network_calls": 0}

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "project": self.project,
                "anonymous_review_forces_off": True,
                "uploads_evidence_by_default": upload_evidence_enabled(),
                "authoritative": "local OTel + canonical artifacts"}
