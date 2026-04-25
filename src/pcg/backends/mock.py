"""
MockBackend — deterministic LLM for smoke tests, CI, and theory unit tests.

Does NOT call any model. Inspects the prompt for telltale patterns
("Question:", "Context:", "Answer:") and returns a templated answer that
either:
    (a) extracts a span from the context that matches a question keyword,
    (b) returns a fixed string for known smoke-test patterns, or
    (c) returns "I don't know." as a safe default.

Why so simple? Because the FRAMEWORK is what we're testing in smoke runs:
the certificate machinery, the checker, the responsibility estimator, the
risk policy. We don't need an actual LLM to verify the plumbing is correct.
A real LLM is only needed for measuring real-world utility, which is what
R1-R5 do with HFLocalBackend.

The mock is bit-deterministic given (prompt, seed): the same call twice
returns identical output. This is what makes replay tests work without an
LLM at all.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

from pcg.backends.base import GenerationOutput
from pcg.eval.meter import count_tokens


# Patterns the mock recognizes
_QUESTION_RE = re.compile(r"Question:\s*(.+?)(?:\n|$)", flags=re.IGNORECASE)
_CONTEXT_RE = re.compile(r"Context:\s*(.+?)(?=\n[A-Z][a-z]+:|\Z)", flags=re.IGNORECASE | re.DOTALL)


def _extract_answer_span(question: str, context: str) -> str | None:
    """A naive rule that finds a span in `context` that contains the
    most question content words.

    Splits context into sentences, scores each by question-word overlap,
    returns the top sentence. This is intentionally a simple heuristic;
    its purpose is to give the framework a non-trivial signal to operate
    on (some right, some wrong) without a real LLM.
    """
    if not context.strip():
        return None
    q_words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 3}
    if not q_words:
        return None
    sents = re.split(r"(?<=[.!?])\s+", context.strip())
    best_sent = ""
    best_score = -1
    for s in sents:
        s_words = set(re.findall(r"\w+", s.lower()))
        score = len(q_words & s_words)
        if score > best_score:
            best_score = score
            best_sent = s
    return best_sent if best_score > 0 else None


@dataclass
class MockBackend:
    """Deterministic mock LLM. Always usable, never calls a real model."""

    name: str = "mock-llm"
    base_latency_ms: float = 1.0
    fail_rate: float = 0.0   # if > 0, deterministically "fails" some inputs

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        seed: int = 0,
    ) -> GenerationOutput:
        t0 = time.perf_counter()

        # Deterministic "failure" injection driven by prompt hash + seed.
        if self.fail_rate > 0:
            h = int(hashlib.sha256(f"{prompt}|{seed}".encode("utf-8")).hexdigest()[:8], 16)
            if (h % 1000) / 1000.0 < self.fail_rate:
                latency = (time.perf_counter() - t0) * 1000.0 + self.base_latency_ms
                return GenerationOutput(
                    text="I don't know.",
                    tokens_in=count_tokens(prompt),
                    tokens_out=count_tokens("I don't know."),
                    latency_ms=latency,
                    finish="stop",
                    backend=self.name,
                    meta={"injected_failure": True, "seed": seed},
                )

        q_match = _QUESTION_RE.search(prompt)
        c_match = _CONTEXT_RE.search(prompt)
        question = q_match.group(1).strip() if q_match else ""
        context = c_match.group(1).strip() if c_match else ""

        # Fixed responses for known smoke-test patterns
        if "calc_tool_log" in prompt.lower():
            # Pull the "X = Y" pattern from the calc tool log and return Y
            m = re.search(r"=\s*([\-\d\.]+)", prompt)
            if m:
                answer = m.group(1)
                latency = (time.perf_counter() - t0) * 1000.0 + self.base_latency_ms
                return GenerationOutput(
                    text=answer, tokens_in=count_tokens(prompt),
                    tokens_out=count_tokens(answer), latency_ms=latency,
                    finish="stop", backend=self.name, meta={"seed": seed},
                )

        # Default: extract a span from context
        span = _extract_answer_span(question, context) if (question and context) else None
        if span is None:
            answer = "I don't know."
        else:
            # Truncate to a short answer-ish length
            answer = span[: max(30, max_tokens * 4)]

        latency = (time.perf_counter() - t0) * 1000.0 + self.base_latency_ms
        return GenerationOutput(
            text=answer,
            tokens_in=count_tokens(prompt),
            tokens_out=count_tokens(answer),
            latency_ms=latency,
            finish="stop",
            backend=self.name,
            meta={"seed": seed},
        )

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)
