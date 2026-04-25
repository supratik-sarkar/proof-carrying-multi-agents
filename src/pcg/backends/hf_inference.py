"""
HFInferenceBackend — call HF Inference API endpoints.

This is what unlocks frontier-class models (Llama-3.3-70B, Mixtral-8x22B,
DeepSeek-67B) on a laptop without paying. Free tier is rate-limited; we
cache results aggressively so re-running the same experiment doesn't
re-incur API calls.

Auth: set HF_TOKEN env var to your HuggingFace access token. Generate one
at https://huggingface.co/settings/tokens with "Read" permission.

Determinism: HF Inference API does NOT guarantee bit-stable output even at
temperature 0 — they may A/B route to different model copies. We mark all
HFInferenceBackend outputs as "best-effort deterministic" in their meta;
the Verifier knows to only use cached outputs for replay.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcg.backends.base import GenerationOutput


@dataclass
class HFInferenceBackend:
    """Remote calls to HF Inference Endpoints.

    Args:
        model_name: e.g. "meta-llama/Llama-3.3-70B-Instruct"
        token: HF access token. Defaults to env var HF_TOKEN.
        cache_dir: Path for cached responses (keyed on prompt+seed+params).
        timeout_s: per-request timeout.
    """

    model_name: str
    name: str = ""
    token: str | None = None
    cache_dir: str | None = None
    timeout_s: float = 60.0
    base_url: str = "https://api-inference.huggingface.co/models/"
    _cache_path: Path | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model_name
        if self.token is None:
            self.token = os.environ.get("HF_TOKEN")
        if self.cache_dir:
            self._cache_path = Path(self.cache_dir)
            self._cache_path.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prompt: str, params: dict[str, Any]) -> str:
        import hashlib
        blob = json.dumps({"p": prompt, **params, "m": self.model_name}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _try_cache_load(self, key: str) -> GenerationOutput | None:
        if self._cache_path is None:
            return None
        f = self._cache_path / f"{key}.json"
        if not f.exists():
            return None
        try:
            d = json.loads(f.read_text())
            return GenerationOutput(**d)
        except Exception:
            return None

    def _cache_store(self, key: str, out: GenerationOutput) -> None:
        if self._cache_path is None:
            return
        f = self._cache_path / f"{key}.json"
        f.write_text(json.dumps({
            "text": out.text, "tokens_in": out.tokens_in,
            "tokens_out": out.tokens_out, "latency_ms": out.latency_ms,
            "finish": out.finish, "backend": out.backend, "meta": out.meta,
        }))

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
        params = {"max_tokens": max_tokens, "temperature": temperature,
                  "top_p": top_p, "stop": stop, "seed": seed}
        key = self._cache_key(prompt, params)
        cached = self._try_cache_load(key)
        if cached is not None:
            cached.meta["cache_hit"] = True
            return cached

        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "HFInferenceBackend requires `httpx`. "
                "Install with `pip install httpx` (also in [web] extras)."
            ) from exc

        if not self.token:
            raise RuntimeError(
                "HF_TOKEN not set. Set the env var or pass token=... explicitly."
            )

        url = f"{self.base_url}{self.model_name}"
        headers = {"Authorization": f"Bearer {self.token}",
                   "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "temperature": max(temperature, 1e-7),
                "top_p": top_p,
                "do_sample": temperature > 0,
            },
            "options": {"wait_for_model": True, "use_cache": True},
        }

        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
        latency = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            return GenerationOutput(
                text="", tokens_in=0, tokens_out=0, latency_ms=latency,
                finish="error", backend=self.name,
                meta={"http_status": r.status_code, "body": r.text[:500]},
            )

        data = r.json()
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            text = data.get("generated_text", "")
        else:
            text = ""

        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s, 1)[0]

        from pcg.eval.meter import count_tokens
        out = GenerationOutput(
            text=text,
            tokens_in=count_tokens(prompt),
            tokens_out=count_tokens(text),
            latency_ms=latency,
            finish="stop",
            backend=self.name,
            meta={"seed": seed, "cache_hit": False},
        )
        self._cache_store(key, out)
        return out

    def count_tokens(self, text: str) -> int:
        from pcg.eval.meter import count_tokens
        return count_tokens(text)
