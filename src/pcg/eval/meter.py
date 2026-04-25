"""
Overhead meter (R5).

Instrument every LLM call, tool call, retrieval operation, hash computation,
and replay step. Emits per-phase latency, token counts, and operation counts
so that the paper's R5 result is a direct measurement, not an estimate.

The design: all measured operations go through the `Meter` context manager.
Downstream modules (`pcg.agents.*`) accept an optional `meter` kwarg and
instrument the appropriate calls. When no meter is passed, a no-op meter is
used so performance is not degraded in un-instrumented runs.

Usage:
    meter = Meter()
    with meter.phase("retrieval"):
        ...retrieve...
    with meter.phase("llm_gen", tokens_in=123):
        ...generate...
        meter.record_tokens(tokens_out=45)
    report = meter.report()
    print(report.to_table())
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


# -----------------------------------------------------------------------------
# Token counting helpers
# -----------------------------------------------------------------------------


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Fast token count. Uses tiktoken when available, else a word-ish approx.

    For research overhead metrics we don't need exact tokenizer agreement with
    the serving LLM, only consistent accounting across runs. tiktoken's
    cl100k_base (OpenAI ChatGPT) is a reasonable default; HF tokenizers
    typically give counts within ~5% of this for English.
    """
    if _HAS_TIKTOKEN:
        enc = tiktoken.get_encoding(model) if not model.startswith("gpt") \
            else tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    # Fallback: whitespace tokens with subword penalty.
    return int(1.3 * len(text.split()))


# -----------------------------------------------------------------------------
# Phase timer
# -----------------------------------------------------------------------------


@dataclass
class PhaseTimer:
    """One phase's accumulated measurements."""

    name: str
    n_calls: int = 0
    total_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    n_tool_calls: int = 0
    n_hash_ops: int = 0

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "phase": self.name,
            "n_calls": self.n_calls,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.total_ms / self.n_calls, 3) if self.n_calls else 0.0,
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "n_tool_calls": self.n_tool_calls,
            "n_hash_ops": self.n_hash_ops,
        }


# -----------------------------------------------------------------------------
# The Meter
# -----------------------------------------------------------------------------


@dataclass
class MeterReport:
    """Structured report of one Meter's lifetime. JSON-serializable."""

    phases: dict[str, PhaseTimer] = field(default_factory=dict)
    wall_ms: float = 0.0

    def total_tokens(self) -> int:
        return sum(p.total_tokens_in + p.total_tokens_out for p in self.phases.values())

    def total_tool_calls(self) -> int:
        return sum(p.n_tool_calls for p in self.phases.values())

    def total_hash_ops(self) -> int:
        return sum(p.n_hash_ops for p in self.phases.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "wall_ms": round(self.wall_ms, 3),
            "total_tokens": self.total_tokens(),
            "total_tool_calls": self.total_tool_calls(),
            "total_hash_ops": self.total_hash_ops(),
            "phases": [p.to_dict() for p in self.phases.values()],
        }

    def to_table(self) -> str:
        """Pretty-print as a fixed-width table for logging."""
        rows = [
            ("phase", "n", "total_ms", "avg_ms", "tok_in", "tok_out", "tools", "hashes"),
        ]
        for p in self.phases.values():
            rows.append((
                p.name,
                str(p.n_calls),
                f"{p.total_ms:.1f}",
                f"{p.total_ms / p.n_calls:.2f}" if p.n_calls else "-",
                str(p.total_tokens_in),
                str(p.total_tokens_out),
                str(p.n_tool_calls),
                str(p.n_hash_ops),
            ))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        sep = "  ".join("-" * w for w in widths)
        out = []
        for i, r in enumerate(rows):
            out.append("  ".join(r[j].ljust(widths[j]) for j in range(len(r))))
            if i == 0:
                out.append(sep)
        out.append("")
        out.append(f"Wall: {self.wall_ms:.1f} ms  |  Total tokens: {self.total_tokens()}")
        return "\n".join(out)


class Meter:
    """Per-claim meter.

    Create a Meter per claim (or per rollout), wrap instrumented regions in
    `meter.phase("name")`, and call `meter.report()` at the end.

    Thread-safety: this class is NOT thread-safe. Use one Meter per thread.
    """

    def __init__(self) -> None:
        self._phases: dict[str, PhaseTimer] = {}
        self._t0 = time.perf_counter()
        self._current: str | None = None

    @contextmanager
    def phase(
        self,
        name: str,
        tokens_in: int = 0,
    ) -> Iterator["Meter"]:
        """Time a region of code; accumulate into phase `name`.

        Nested phases are supported; timing is per-phase (a nested phase's
        time is counted in both its own and the enclosing phase, which is
        usually what you want — e.g., retrieval_llm inside retrieval).
        """
        pt = self._phases.setdefault(name, PhaseTimer(name=name))
        pt.n_calls += 1
        pt.total_tokens_in += tokens_in
        t0 = time.perf_counter()
        prev = self._current
        self._current = name
        try:
            yield self
        finally:
            pt.total_ms += (time.perf_counter() - t0) * 1000.0
            self._current = prev

    def record_tokens(self, tokens_out: int = 0, tokens_in: int = 0) -> None:
        """Attribute tokens to the currently active phase."""
        if self._current is None:
            return
        pt = self._phases[self._current]
        pt.total_tokens_in += tokens_in
        pt.total_tokens_out += tokens_out

    def record_tool_call(self) -> None:
        if self._current is None:
            return
        self._phases[self._current].n_tool_calls += 1

    def record_hash(self) -> None:
        if self._current is None:
            return
        self._phases[self._current].n_hash_ops += 1

    def report(self) -> MeterReport:
        return MeterReport(
            phases=dict(self._phases),
            wall_ms=(time.perf_counter() - self._t0) * 1000.0,
        )


class NullMeter(Meter):
    """No-op meter for un-instrumented runs. Same API, zero overhead."""

    def __init__(self) -> None:       # noqa: D401 — docstring in parent
        self._phases = {}
        self._t0 = 0.0
        self._current = None

    @contextmanager
    def phase(self, name: str, tokens_in: int = 0) -> Iterator["Meter"]:
        yield self

    def record_tokens(self, tokens_out: int = 0, tokens_in: int = 0) -> None:
        return

    def record_tool_call(self) -> None:
        return

    def record_hash(self) -> None:
        return

    def report(self) -> MeterReport:
        return MeterReport()
