"""Provider abstraction. Every response emits the fields A10 needs.

Routes: offline_mock | local_hf | hosted_provider.
No provider call is made during the remediation pass.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..canon import fingerprint, sha256_text


@dataclass
class ProviderResponse:
    text: str
    model_id: str
    model_revision: str
    provider_route: str
    backend_type: str
    dtype: Optional[str]
    quantization: Optional[str]
    decoding_config: Dict[str, Any]
    seed: Optional[int]
    latency_ms: float
    tokens_in: int
    tokens_out: int
    billed_cost_usd: Optional[float]
    raw_output_sha256: str
    backend_fingerprint: str
    cache_state: str = "cold"

    def to_dict(self) -> dict:
        return vars(self)


class Provider(Protocol):
    route: str
    def generate(self, prompt: str, **kw) -> ProviderResponse: ...


def make_fingerprint(model_id, revision, tokenizer_id, backend_type, route,
                     dtype, quantization, decoding, seed) -> str:
    return fingerprint(model_id=model_id, revision=revision, tokenizer_id=tokenizer_id,
                       backend_type=backend_type, provider_route=route, dtype=dtype,
                       quantization=quantization, decoding=decoding, seed=seed)


def resolve_device(prefer_cuda: bool = True) -> str:
    """CUDA -> MPS -> CPU. No CUDA-only API is touched at import time."""
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    try:
        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
