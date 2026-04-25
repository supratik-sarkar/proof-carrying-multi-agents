"""
LLM backends used by the agent layer.

Three backends, all implementing the `LLMBackend` protocol:

    - `MockBackend`     : deterministic, no model needed (CI, smoke tests)
    - `HFLocalBackend`  : Hugging Face transformers, MPS / CUDA / CPU autodetect
    - `HFInferenceBackend` : remote HF Inference API (free tier rate-limited)

The Prover and Verifier accept any `LLMBackend`, so swapping backends is a
config change. This is what lets us mix open-weight (local) and frontier
(remote) models in the same experiment - directly answering ICML W1.
"""
from __future__ import annotations

from pcg.backends.base import GenerationOutput, LLMBackend
from pcg.backends.mock import MockBackend

__all__ = ["GenerationOutput", "LLMBackend", "MockBackend"]
