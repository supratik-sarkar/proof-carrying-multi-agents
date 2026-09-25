"""Hierarchical OTel execution fabric.

    experiment -> cell -> seed -> example -> system -> graph -> node
                -> provider|retrieval|tool|checker|policy|replay|guardrail

Canonical execution IDs are carried LOCALLY. `pcg.*` baggage is never propagated
to third-party provider endpoints -- only the W3C traceparent crosses that
boundary -- and no secret is ever placed in an attribute or a link.
"""
from __future__ import annotations

import contextlib
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

try:
    from opentelemetry import trace as _ot  # type: ignore
    HAVE_OTEL_SDK = True
except Exception:
    _ot = None
    HAVE_OTEL_SDK = False

LEVELS = ("experiment", "cell", "seed", "example", "system", "graph", "node", "operation")
OP_KINDS = ("provider", "retrieval", "tool", "checker", "policy", "replay", "guardrail")

#: Attributes that must NEVER appear on a span or in baggage.
FORBIDDEN_ATTRS = ("api_key", "authorization", "secret", "token_value",
                   "password", "credential", "bearer")


class SecretInTelemetry(ValueError):
    pass


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    level: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    start_ms: float = 0.0
    duration_ms: Optional[float] = None
    status: str = "UNSET"
    exception: Optional[str] = None

    def to_dict(self) -> dict:
        return vars(self)


class Tracer:
    """Local-first tracer. Works with or without the OTel SDK installed."""

    def __init__(self):
        self._spans: List[Span] = []
        self._stack = threading.local()
        self._lock = threading.Lock()

    # -------------------------------------------------------------- context
    @property
    def stack(self) -> List[Span]:
        if not hasattr(self._stack, "v"):
            self._stack.v = []
        return self._stack.v

    @property
    def current(self) -> Optional[Span]:
        return self.stack[-1] if self.stack else None

    def traceparent(self) -> Optional[str]:
        """W3C traceparent -- the ONLY context that may cross to a third party."""
        cur = self.current
        if cur is None:
            return None
        return f"00-{cur.trace_id}-{cur.span_id}-01"

    # ----------------------------------------------------------------- span
    @contextlib.contextmanager
    def span(self, name: str, level: str = "operation",
             links: Optional[Sequence[Dict[str, str]]] = None, **attrs) -> Iterator[Span]:
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
        for k in attrs:
            if any(f in k.lower() for f in FORBIDDEN_ATTRS):
                raise SecretInTelemetry(f"attribute {k!r} may contain a secret")
        parent = self.current
        sp = Span(
            span_id=secrets.token_hex(8),
            trace_id=parent.trace_id if parent else secrets.token_hex(16),
            parent_id=parent.span_id if parent else None,
            name=name, level=level, attributes=dict(attrs),
            links=[dict(l) for l in (links or [])],
            start_ms=time.time() * 1000.0)
        self.stack.append(sp)
        t0 = time.perf_counter()
        try:
            yield sp
            sp.status = "OK"
        except BaseException as e:
            sp.status = "ERROR"
            sp.exception = f"{type(e).__name__}: {e}"
            sp.events.append({"name": "exception", "type": type(e).__name__})
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - t0) * 1000.0
            self.stack.pop()
            with self._lock:
                self._spans.append(sp)

    def event(self, name: str, **fields) -> None:
        cur = self.current
        if cur is not None:
            cur.events.append({"name": name, **fields})

    def link_to(self, span_id: str, trace_id: str, relation: str) -> Dict[str, str]:
        """Span LINK, not parentage: a replay is not a causal child of the original."""
        return {"linked_span_id": span_id, "linked_trace_id": trace_id,
                "relation": relation}

    # ---------------------------------------------------------------- export
    def spans(self) -> List[Span]:
        with self._lock:
            return list(self._spans)

    def reset(self) -> None:
        with self._lock:
            self._spans.clear()

    def tree(self) -> Dict[str, List[str]]:
        kids: Dict[str, List[str]] = {}
        for s in self.spans():
            kids.setdefault(s.parent_id or "ROOT", []).append(s.span_id)
        return kids

    def status(self) -> Dict[str, Any]:
        return {"otel_sdk_installed": HAVE_OTEL_SDK, "local_tracer": True,
                "spans": len(self.spans()), "levels": list(LEVELS),
                "operation_kinds": list(OP_KINDS)}


TRACER = Tracer()

#: Required attribute names (semantic-convention aligned where one exists).
ATTR = {
    "experiment_id": "pcg.experiment_id", "run_id": "pcg.run_id",
    "cell_id": "pcg.cell_id", "example_id": "pcg.example_id",
    "dataset": "pcg.dataset", "split": "pcg.split", "system": "pcg.system",
    "seed": "pcg.seed", "spec_hash": "pcg.spec_hash", "prompt_hash": "pcg.prompt_hash",
    "policy_bundle_hash": "pcg.policy_bundle_hash", "checker_hash": "pcg.checker_hash",
    "certificate_root": "pcg.certificate_root", "execution_mode": "pcg.execution_mode",
    "provider": "gen_ai.provider.name", "request_model": "gen_ai.request.model",
    "response_model": "gen_ai.response.model",
    "input_tokens": "gen_ai.usage.input_tokens", "output_tokens": "gen_ai.usage.output_tokens",
    "reasoning_tokens": "pcg.reasoning_tokens", "cached_tokens": "pcg.cached_tokens",
    "retry_attempt": "pcg.retry_attempt", "retry_trigger": "pcg.retry_trigger_class",
    "cache_hit": "pcg.cache_hit", "cost_usd": "pcg.cost_usd",
    "v_h": "pcg.V_H", "v_pi": "pcg.V_Pi", "v_gamma": "pcg.V_Gamma",
    "v_entail": "pcg.V_entail", "accepted": "pcg.accepted",
    "controller_action": "pcg.controller_action",
    "policy_eval_status": "pcg.policy_eval_status",
}


def safe_outbound_headers(tracer: Tracer = TRACER) -> Dict[str, str]:
    """Headers for a third-party call: traceparent ONLY, never pcg.* baggage."""
    tp = tracer.traceparent()
    return {"traceparent": tp} if tp else {}
