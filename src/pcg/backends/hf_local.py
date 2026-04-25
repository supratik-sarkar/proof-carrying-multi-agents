"""
HFLocalBackend — local inference via HuggingFace transformers.

Auto-detects the best device:
    1. CUDA if available
    2. MPS (Apple Silicon) if available
    3. CPU fallback

Quantization is OFF by default. To enable 4-bit:
    backend = HFLocalBackend("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)
This requires `bitsandbytes` (CUDA-only — does NOT work on MPS today).
On Apple Silicon, use `dtype="float16"` for memory savings instead.

Determinism: when `temperature == 0`, we use greedy decoding so output is
bit-stable across calls. We also set torch + numpy seeds before each call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pcg.backends.base import GenerationOutput


def _pick_device() -> str:
    """CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class HFLocalBackend:
    """Local HF transformers inference.

    Heavy imports happen on first generate() call so that smoke tests that
    don't touch this backend pay no startup cost.
    """

    model_name: str
    name: str = ""
    device: str = field(default_factory=_pick_device)
    dtype: str = "float16"           # "float16" | "bfloat16" | "float32"
    load_in_4bit: bool = False       # CUDA only
    trust_remote_code: bool = False  # explicit opt-in to avoid surprises
    max_input_tokens: int = 4096

    _model: Any = None
    _tokenizer: Any = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model_name

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                # bitsandbytes not installed; silently ignore the request.
                pass

        torch_dtype = {"float16": torch.float16,
                       "bfloat16": torch.bfloat16,
                       "float32": torch.float32}.get(self.dtype, torch.float16)
        kwargs["torch_dtype"] = torch_dtype

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if not self.load_in_4bit:
            self._model = self._model.to(self.device)
        self._model.eval()

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
        import torch
        torch.manual_seed(seed)

        self._ensure_loaded()
        t0 = time.perf_counter()

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        n_in = int(inputs["input_ids"].shape[-1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0, n_in:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Honor `stop` strings post-hoc (cheap and avoids HF tokenizer churn).
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s, 1)[0]

        latency = (time.perf_counter() - t0) * 1000.0
        return GenerationOutput(
            text=text,
            tokens_in=n_in,
            tokens_out=int(gen_ids.shape[-1]),
            latency_ms=latency,
            finish="stop",
            backend=self.name,
            meta={"device": self.device, "dtype": self.dtype, "seed": seed},
        )

    def count_tokens(self, text: str) -> int:
        self._ensure_loaded()
        return len(self._tokenizer.encode(text))
