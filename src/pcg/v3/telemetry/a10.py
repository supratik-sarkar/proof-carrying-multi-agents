"""A10 cost/latency aggregation directly from trace artifacts.

A10 must not reconstruct timing or cost from unrelated print logs: it reads the
same spans the UI reads. Latency is reported OBSERVED with the instrumentation
overhead reported separately -- never silently subtracted.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .hierarchy import ATTR, OP_KINDS, Span


def _pct(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))]


def _rate(num, den):
    return None if not den else num / den          # undefined -> None, never 0.0


def aggregate_a10_from_traces(spans: Iterable[Span],
                              overhead: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Aggregate per-system cost/latency from spans alone."""
    spans = list(spans)
    by_system: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "graph_latency_ms": [], "tokens_in": 0, "tokens_out": 0,
        "cost_usd": 0.0, "accepted": 0, "accepted_correct": 0, "runs": 0,
        "calls": {k: 0 for k in OP_KINDS}, "retries": 0,
        "phase_ms": defaultdict(float),
    })
    for s in spans:
        sysname = s.attributes.get(ATTR["system"]) or "unknown"
        b = by_system[sysname]
        if s.level == "graph":
            b["runs"] += 1
            if s.duration_ms is not None:
                b["graph_latency_ms"].append(s.duration_ms)
            if s.attributes.get(ATTR["accepted"]) is True:
                b["accepted"] += 1
                if s.attributes.get("pcg.h_joint") in (0, 0.0, False, None):
                    b["accepted_correct"] += 1
        if s.level == "node" and s.duration_ms is not None:
            b["phase_ms"][s.name] += s.duration_ms
        kind = s.attributes.get("pcg.op_kind")
        if kind in OP_KINDS:
            b["calls"][kind] += 1
        b["tokens_in"] += int(s.attributes.get(ATTR["input_tokens"]) or 0)
        b["tokens_out"] += int(s.attributes.get(ATTR["output_tokens"]) or 0)
        b["cost_usd"] += float(s.attributes.get(ATTR["cost_usd"]) or 0.0)
        if s.attributes.get(ATTR["retry_attempt"]):
            b["retries"] += 1

    out: Dict[str, Any] = {}
    for sysname, b in by_system.items():
        lat = b["graph_latency_ms"]
        out[sysname] = {
            "runs": b["runs"],
            "latency_p50_ms_observed": _pct(lat, 0.50),
            "latency_p95_ms_observed": _pct(lat, 0.95),
            "tokens_in": b["tokens_in"], "tokens_out": b["tokens_out"],
            "model_calls": b["calls"]["provider"],
            "retrieval_calls": b["calls"]["retrieval"],
            "tool_calls": b["calls"]["tool"],
            "checker_calls": b["calls"]["checker"],
            "replay_calls": b["calls"]["replay"],
            "policy_calls": b["calls"]["policy"],
            "guardrail_calls": b["calls"]["guardrail"],
            "physical_retries": b["retries"],
            "cost_usd": b["cost_usd"],
            "accepted": b["accepted"], "accepted_correct": b["accepted_correct"],
            "cost_per_accepted_correct": _rate(b["cost_usd"], b["accepted_correct"]),
            "phase_ms": dict(b["phase_ms"]),
        }
    payload: Dict[str, Any] = {"by_system": out, "source": "otel_spans",
                               "note": "latency is OBSERVED; overhead reported separately"}
    if overhead:
        payload["instrumentation"] = {
            "overhead_ms": overhead.get("delta_mean_ms"),
            "overhead_ratio": overhead.get("ratio"),
            "tau_instr": overhead.get("tau_instr"),
            "instrumentation_limited": overhead.get("instrumentation_limited"),
            "uncorrected_fields": overhead.get("uncorrected_fields"),
        }
    return payload
