"""
Common backend types.

`LLMBackend` is a Protocol, not an ABC, so duck-typed implementations work.
All backends speak the same minimal surface:

    backend.generate(prompt, **opts) -> GenerationOutput

Tool-use is NOT routed through the backend interface: the Prover handles
tool dispatch in Python. This keeps the backend interface narrow and means
we can switch a Prover from "local Llama" to "remote 70B" without touching
tool code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationOutput:
    """Result of one LLM call.

    Fields:
        text:        the generated text (post-decoding, no special tokens)
        tokens_in:   number of input tokens charged
        tokens_out:  number of generated tokens
        latency_ms:  wall-clock time of the call
        logprobs:    optional, per-output-token log-probabilities
        finish:      "stop" | "length" | "error"
        backend:     identifier string for the backend that produced this
        meta:        free-form (model name, sampling params, request id, ...)
    """

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    logprobs: list[float] | None = None
    finish: str = "stop"
    backend: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class LLMBackend(Protocol):
    """Minimal LLM interface.

    Implementations must be deterministic when `temperature == 0` AND `seed`
    is provided. This is required for the `replay` contract to hold —
    Verifier replays must produce bit-identical outputs.
    """

    name: str

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        seed: int = 0,
    ) -> GenerationOutput: ...

    def count_tokens(self, text: str) -> int: ...
