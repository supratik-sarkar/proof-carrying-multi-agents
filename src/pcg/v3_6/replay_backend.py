"""Strict, non-network replay backend.

Exists solely because the prover architecture expects an LLMBackend. It never
generates and never touches the network. It satisfies the interface by handing
back text that the v2.3.2 dispatcher already produced, and it refuses unless
the prompt the prover builds hashes identically to the generation prompt.
"""
from typing import Any, Dict, Optional

from pcg.backends.base import GenerationOutput

from .hashing36 import sha256_text

CONFIDENCE_SOURCE = "RETRIEVAL_SCORE_FALLBACK_NO_PROVIDER_LOGPROBS"


class PromptIdentityMismatch(RuntimeError):
    pass


class ReplayViolation(RuntimeError):
    pass


class ReplayBackend:
    """LLMBackend that replays one recorded generation. Zero network calls."""

    name = "replay-v3.6"

    def __init__(self, *, expected_prompt_sha256: str, text: str,
                 tokens_in: int = 0, tokens_out: int = 0, finish: str = "stop",
                 provider: str = "", api_model_id: str = "",
                 meta: Optional[Dict[str, Any]] = None) -> None:
        if not expected_prompt_sha256 or len(expected_prompt_sha256) != 64:
            raise ReplayViolation("EXPECTED_PROMPT_SHA_INVALID")
        if text is None:
            raise ReplayViolation("REPLAY_TEXT_NONE")
        if text == "":
            # A provider that returned nothing is a failed task, never an
            # accepted candidate. Refuse before the object can be used.
            raise ReplayViolation("EMPTY_GENERATION_TEXT_NOT_ACCEPTABLE")
        self.expected_prompt_sha256 = expected_prompt_sha256
        self._text = text
        self._tokens_in = int(tokens_in)
        self._tokens_out = int(tokens_out)
        self._finish = finish
        self._provider = provider
        self._api_model_id = api_model_id
        self._meta = dict(meta or {})
        self.calls = 0
        self.network_calls = 0

    def generate(self, prompt: str, **kwargs: Any) -> GenerationOutput:
        got = sha256_text(prompt)
        if got != self.expected_prompt_sha256:
            raise PromptIdentityMismatch(
                f"PROMPT_SHA_MISMATCH expected={self.expected_prompt_sha256} got={got}")
        self.calls += 1
        if self.calls > 1:
            raise ReplayViolation("MULTIPLE_GENERATE_CALLS_IN_REPLAY")
        meta = dict(self._meta)
        meta.update({
            "replayed": True,
            "provider": self._provider,
            "api_model_id": self._api_model_id,
            "confidence_source": CONFIDENCE_SOURCE,
            "provider_logprobs_available": False,
        })
        return GenerationOutput(
            text=self._text, tokens_in=self._tokens_in, tokens_out=self._tokens_out,
            latency_ms=0.0, logprobs=None, finish=self._finish,
            backend=f"replay::{self._provider}::{self._api_model_id}", meta=meta,
        )
