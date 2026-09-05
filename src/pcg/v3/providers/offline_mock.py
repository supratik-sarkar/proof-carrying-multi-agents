"""Deterministic offline provider. The ONLY provider usable in this pass."""
from __future__ import annotations
import hashlib, time
from typing import Any, Dict, Optional

from ..canon import sha256_text
from .base import ProviderResponse, make_fingerprint


class OfflineMockProvider:
    route = "offline_mock"
    MODEL_ID = "fixture/deterministic"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def generate(self, prompt: str, **kw) -> ProviderResponse:
        t0 = time.perf_counter()
        h = hashlib.sha256(f"{self.seed}:{prompt}".encode()).hexdigest()
        text = f"[offline-mock:{h[:12]}]"
        dt = (time.perf_counter() - t0) * 1000.0
        dec = {"temperature": 0.0, "max_tokens": kw.get("max_tokens", 256)}
        return ProviderResponse(
            text=text, model_id=self.MODEL_ID, model_revision="0",
            provider_route=self.route, backend_type="MOCK", dtype="float32",
            quantization=None, decoding_config=dec, seed=self.seed,
            latency_ms=round(dt, 4), tokens_in=len(prompt.split()),
            tokens_out=len(text.split()), billed_cost_usd=0.0,
            raw_output_sha256=sha256_text(text),
            backend_fingerprint=make_fingerprint(self.MODEL_ID, "0", "fixture", "MOCK",
                                                 self.route, "float32", None, dec, self.seed))
