"""
Operational rigor: latency, token, and cost-under-load analysis.

The R5 overhead figure already shows mean tokens per phase per backend.
This module adds the distributional view that deployment-leaning
reviewers expect:
    - P50 / P95 / P99 latency per claim
    - Token-count distributions per phase
    - Cost-vs-load curves (concurrency → throughput → tail latency)

Designed to consume two input shapes:
    (a) Per-claim sample lists. The richer signal — every claim's
        end-to-end latency, prove tokens, verify tokens, redundant
        tokens. Run scripts capture these into `meter_samples.jsonl`.
    (b) Aggregated MeterReport summaries. Coarser; only enough to
        reconstruct mean and total. Falls back to (b) when (a) absent.

Usage:
    from pcg.eval.latency import (
        load_per_claim_samples, summary_quantiles, cost_curve,
    )
    samples = load_per_claim_samples(Path("results/r5-2026-XX-XX-..."))
    q = summary_quantiles(samples, field="total_ms")
    print(q.p50, q.p95, q.p99)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClaimSample:
    """One claim's full operational measurement."""

    claim_id: str
    backend: str
    total_ms: float                    # end-to-end latency
    prove_ms: float = 0.0
    verify_ms: float = 0.0
    redundant_ms: float = 0.0
    audit_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    n_tool_calls: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimSample":
        return cls(
            claim_id=str(d.get("claim_id", "")),
            backend=str(d.get("backend", "unknown")),
            total_ms=float(d.get("total_ms", 0.0)),
            prove_ms=float(d.get("prove_ms", 0.0)),
            verify_ms=float(d.get("verify_ms", 0.0)),
            redundant_ms=float(d.get("redundant_ms", 0.0)),
            audit_ms=float(d.get("audit_ms", 0.0)),
            tokens_in=int(d.get("tokens_in", 0)),
            tokens_out=int(d.get("tokens_out", 0)),
            n_tool_calls=int(d.get("n_tool_calls", 0)),
        )


@dataclass
class Quantiles:
    """Summary stats over a sample list."""

    field_name: str
    n: int
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    p_max: float

    def to_dict(self) -> dict:
        return {
            "field": self.field_name, "n": self.n,
            "mean": self.mean, "std": self.std,
            "p50": self.p50, "p95": self.p95,
            "p99": self.p99, "p_max": self.p_max,
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_per_claim_samples(run_dir: Path) -> list[ClaimSample]:
    """Load raw per-claim samples if the run captured them.

    Run scripts can dump these into `meter_samples.jsonl` (one JSON
    object per line). Returns empty list if not present, so the caller
    can degrade gracefully to aggregated metrics.
    """
    f = Path(run_dir) / "meter_samples.jsonl"
    if not f.exists():
        return []
    samples: list[ClaimSample] = []
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(ClaimSample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue   # tolerate malformed rows; they're rare
    return samples


# ---------------------------------------------------------------------------
# Quantile computation
# ---------------------------------------------------------------------------


def summary_quantiles(
    samples: Iterable[ClaimSample], field: str = "total_ms",
) -> Quantiles:
    """P50/P95/P99 + mean/std/max for the requested numeric field."""
    arr = np.array([getattr(s, field) for s in samples], dtype=float)
    if arr.size == 0:
        return Quantiles(field_name=field, n=0, mean=0, std=0,
                         p50=0, p95=0, p99=0, p_max=0)
    return Quantiles(
        field_name=field,
        n=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        p50=float(np.quantile(arr, 0.50)),
        p95=float(np.quantile(arr, 0.95)),
        p99=float(np.quantile(arr, 0.99)),
        p_max=float(arr.max()),
    )


def per_backend_quantiles(
    samples: Iterable[ClaimSample], field: str = "total_ms",
) -> dict[str, Quantiles]:
    """Group samples by backend, then compute Quantiles per group."""
    groups: dict[str, list[ClaimSample]] = {}
    for s in samples:
        groups.setdefault(s.backend, []).append(s)
    return {b: summary_quantiles(g, field=field) for b, g in groups.items()}


# ---------------------------------------------------------------------------
# Cost curve under simulated load
# ---------------------------------------------------------------------------


@dataclass
class LoadPoint:
    """One point on the throughput-vs-latency curve."""

    concurrency: int                   # simultaneous in-flight claims
    throughput_per_sec: float          # claims completed per second
    p50_ms: float
    p95_ms: float
    p99_ms: float
    cost_per_claim_usd: float          # $/claim at this load


@dataclass
class CostCurve:
    """A full load sweep — one curve per backend."""

    backend: str
    points: list[LoadPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "points": [p.__dict__ for p in self.points],
        }


def cost_curve(
    samples: list[ClaimSample],
    *,
    cost_per_1k_tokens_in: float = 0.0,
    cost_per_1k_tokens_out: float = 0.0,
    concurrency_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    seed: int = 0,
) -> CostCurve:
    """Simulate latency+throughput under M/M/c-style queueing using the
    measured per-claim distribution as the service-time distribution.

    Real production load is hard to predict from a single-claim trace, so
    we use the empirical service-time distribution to bootstrap a queue
    simulation. Honest version of "what would this look like at scale?"

    Args:
        samples: per-claim measurements (must share a backend; group first).
        cost_per_1k_tokens_in/out: provider pricing. 0 means free tier.
        concurrency_levels: how many simultaneous workers to simulate.

    Returns CostCurve with one LoadPoint per concurrency level.
    """
    if not samples:
        return CostCurve(backend="(empty)")

    backend = samples[0].backend
    service_ms = np.array([s.total_ms for s in samples], dtype=float)
    tokens_in_per = np.array([s.tokens_in for s in samples], dtype=float).mean()
    tokens_out_per = np.array([s.tokens_out for s in samples], dtype=float).mean()
    cost_per_claim = (
        cost_per_1k_tokens_in * tokens_in_per / 1000.0
        + cost_per_1k_tokens_out * tokens_out_per / 1000.0
    )

    rng = np.random.default_rng(seed)
    points: list[LoadPoint] = []
    sim_n = 5_000   # claims to simulate per concurrency level

    for c in concurrency_levels:
        # Single FIFO queue with c workers
        # Treat arrivals as Poisson at λ chosen so utilization ρ=0.7
        mu = c / (service_ms.mean() / 1000.0)   # claims/sec capacity
        lam = 0.7 * mu
        arrivals = np.cumsum(rng.exponential(1.0 / lam, sim_n))   # seconds
        services = rng.choice(service_ms, sim_n) / 1000.0          # seconds
        # Workers stored as next-free-time
        worker_free = np.zeros(c)
        completes = np.empty(sim_n)
        for i in range(sim_n):
            j = int(np.argmin(worker_free))
            start = max(arrivals[i], worker_free[j])
            completes[i] = start + services[i]
            worker_free[j] = completes[i]
        latencies_ms = (completes - arrivals) * 1000.0
        # Throughput = total claims / total wall time
        wall_sec = float(completes[-1])
        throughput = sim_n / wall_sec if wall_sec > 0 else 0.0
        points.append(LoadPoint(
            concurrency=c,
            throughput_per_sec=throughput,
            p50_ms=float(np.quantile(latencies_ms, 0.50)),
            p95_ms=float(np.quantile(latencies_ms, 0.95)),
            p99_ms=float(np.quantile(latencies_ms, 0.99)),
            cost_per_claim_usd=cost_per_claim,
        ))
    return CostCurve(backend=backend, points=points)
