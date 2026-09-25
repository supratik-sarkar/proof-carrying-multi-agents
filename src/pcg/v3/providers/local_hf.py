"""Local Hugging Face route (Mac MPS / Colab CUDA). Lazy: nothing loads on import."""
from __future__ import annotations
import time
from typing import Any, Dict, Optional

from ..canon import sha256_text
from .base import ProviderResponse, make_fingerprint, resolve_device


class LocalHFProvider:
    route = "local_hf"

    def __init__(self, model_id: str, revision: str = "main", dtype: str = "auto",
                 quantization: Optional[str] = None, seed: int = 0):
        self.model_id, self.revision = model_id, revision
        self.dtype, self.quantization, self.seed = dtype, quantization, seed
        self.device = resolve_device()
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        load_kwargs = {}
        if self.dtype in ("bfloat16", "torch.bfloat16"):
            load_kwargs["dtype"] = torch.bfloat16
        elif self.dtype in ("float16", "torch.float16"):
            load_kwargs["dtype"] = torch.float16
        elif self.dtype == "auto":
            load_kwargs["dtype"] = "auto"
        self._tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, **load_kwargs).to(self.device)

    def generate(self, prompt: str, max_tokens: int = 256, **kw) -> ProviderResponse:
        self._load()
        import torch  # type: ignore
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.manual_seed(self.seed)
        t0 = time.perf_counter()
        enc = self._tok(prompt, return_tensors="pt").to(self.device)
        out = self._model.generate(**enc, max_new_tokens=max_tokens, do_sample=False)
        text = self._tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        dt = (time.perf_counter() - t0) * 1000.0
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        dec = {
            "do_sample": False,
            "max_tokens": max_tokens,
            "temperature": "NOT_APPLICABLE",
            "top_p": "NOT_APPLICABLE",
        }
        return ProviderResponse(
            text=text, model_id=self.model_id, model_revision=self.revision,
            provider_route=self.route, backend_type="LOCAL_MODEL", dtype=self.dtype,
            quantization=self.quantization, decoding_config=dec, seed=self.seed,
            latency_ms=round(dt, 3), tokens_in=int(enc["input_ids"].shape[1]),
            tokens_out=int(out.shape[1] - enc["input_ids"].shape[1]),
            billed_cost_usd=None, raw_output_sha256=sha256_text(text),
            backend_fingerprint=make_fingerprint(self.model_id, self.revision,
                                                 self.model_id, "LOCAL_MODEL", self.route,
                                                 self.dtype, self.quantization, dec, self.seed))
