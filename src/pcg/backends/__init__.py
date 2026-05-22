"""
LLM backends used by the PCG-MAS agent layer.

All backends implement the `LLMBackend` protocol:

    - `MockBackend`        : deterministic offline backend for preflight and CI
    - `HFLocalBackend`     : Hugging Face transformers with local device support
    - `HFInferenceBackend` : remote Hugging Face Inference API / endpoint backend

The Prover and Verifier accept any `LLMBackend`, so swapping local, offline,
and remote model execution is a configuration-level change.
"""
from __future__ import annotations

from pcg.backends.base import GenerationOutput, LLMBackend
from pcg.backends.mock import MockBackend

try:
    from pcg.backends.hf_local import HFLocalBackend
except Exception:  # optional dependency path
    HFLocalBackend = None  # type: ignore

try:
    from pcg.backends.hf_inference import HFInferenceBackend
except Exception:  # optional dependency path
    HFInferenceBackend = None  # type: ignore

__all__ = [
    "GenerationOutput",
    "LLMBackend",
    "MockBackend",
    "HFLocalBackend",
    "HFInferenceBackend",
]
