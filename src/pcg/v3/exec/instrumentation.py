"""Paired instrumentation-overhead calibration (D3).

Measured latency is  T_obs = T_true + Delta_instr.  With per-node, per-provider,
per-checker, per-policy and per-guardrail spans, Delta_instr is not negligible,
so A10 would partly be measuring the instrument.

Convention (frozen): report OBSERVED end-to-end latency, and report estimated
instrumentation overhead SEPARATELY. Do not blindly subtract it. If

    Delta_hat / T_true_hat > tau_instr        (tau_instr = 0.05)

the latency result is marked INSTRUMENTATION_LIMITED.

Overhead is decomposed into span construction/local recording and exporter cost,
and is measured in the instrumentation mode intended for the final experiment.
Token counts, provider/tool/retrieval call counts and billed dollars are NOT
corrected by this calibration -- only latency is contaminated.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

TAU_INSTR = 0.05


class InstrumentationMode(str, Enum):
    OFF = "OFF"                    # no spans constructed
    LOCAL_ONLY = "LOCAL_ONLY"      # spans constructed + recorded locally, no exporter
    EXPORTING = "EXPORTING"        # spans constructed + exported


@dataclass
class OverheadEstimate:
    mode: InstrumentationMode
    n: int
    t_obs_mean_ms: float
    t_true_mean_ms: float
    delta_mean_ms: float
    delta_ci95_ms: List[float]
    span_construction_ms: float
    exporter_ms: float
    ratio: Optional[float]
    tau_instr: float = TAU_INSTR
    instrumentation_limited: bool = False
    corrected_fields: List[str] = field(default_factory=lambda: ["latency"])
    uncorrected_fields: List[str] = field(default_factory=lambda: [
        "tokens_in", "tokens_out", "model_calls", "retrieval_calls",
        "tool_calls", "checker_calls", "replay_calls", "billed_cost_usd"])

    def to_dict(self) -> dict:
        return {**vars(self), "mode": self.mode.value}


def _ci95(xs: List[float]) -> List[float]:
    if len(xs) < 2:
        return [float("nan"), float("nan")]
    m = statistics.fmean(xs)
    se = statistics.stdev(xs) / (len(xs) ** 0.5)
    return [m - 1.96 * se, m + 1.96 * se]


def measure_overhead(workload: Callable[[], None],
                     instrumented: Callable[[], None],
                     exporting: Optional[Callable[[], None]] = None,
                     n: int = 200,
                     mode: InstrumentationMode = InstrumentationMode.LOCAL_ONLY,
                     tau: float = TAU_INSTR) -> OverheadEstimate:
    """Paired A/B on an identical deterministic workload. No provider calls."""
    def timed(fn: Callable[[], None]) -> List[float]:
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t0) * 1000.0)
        return out

    # interleave to cancel drift
    base: List[float] = []
    inst: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter(); workload(); base.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter(); instrumented(); inst.append((time.perf_counter() - t1) * 1000.0)

    exp_ms = 0.0
    if exporting is not None:
        e = timed(exporting)
        exp_ms = max(0.0, statistics.fmean(e) - statistics.fmean(inst))

    t_true = statistics.fmean(base)
    t_obs = statistics.fmean(inst) + exp_ms
    deltas = [i - b for i, b in zip(inst, base)]
    delta = max(0.0, t_obs - t_true)
    ratio = (delta / t_true) if t_true > 0 else None
    return OverheadEstimate(
        mode=mode, n=n, t_obs_mean_ms=t_obs, t_true_mean_ms=t_true,
        delta_mean_ms=delta, delta_ci95_ms=_ci95(deltas),
        span_construction_ms=max(0.0, statistics.fmean(inst) - t_true),
        exporter_ms=exp_ms, ratio=ratio,
        instrumentation_limited=bool(ratio is not None and ratio > tau), tau_instr=tau)


def annotate_latency(latency_ms: Optional[float], est: OverheadEstimate) -> Dict[str, object]:
    """Attach the calibration to a reported latency WITHOUT silently subtracting."""
    return {
        "latency_ms_observed": latency_ms,
        "instrumentation_overhead_ms": est.delta_mean_ms,
        "instrumentation_overhead_ratio": est.ratio,
        "instrumentation_mode": est.mode.value,
        "status": "INSTRUMENTATION_LIMITED" if est.instrumentation_limited else "OK",
        "note": ("observed latency is reported as measured; overhead is reported "
                 "separately and is not subtracted"),
    }
