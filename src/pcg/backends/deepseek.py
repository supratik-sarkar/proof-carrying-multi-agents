"""
DeepSeek official API backend (platform.deepseek.com, OpenAI-compatible).

Used for deepseek-v3 (671B MoE) which cannot run locally and is NOT served by
HF's serverless inference. Token resolved from DEEPSEEK_API_KEY env only;
never read from source-controlled files, printed, or written into artifacts.

Determinism: the DeepSeek API does not guarantee bit-identical outputs even at
temperature=0, so we cache every (prompt, params, seed) -> output on disk. The
Verifier replay then reads the cached output, satisfying the replay contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcg.backends.base import GenerationOutput

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# DeepSeek-V3 is served under this chat model id on the official API.
_DEEPSEEK_CHAT_MODEL = "deepseek-chat"


@dataclass
class DeepSeekBackend:
    model_name: str = "deepseek-ai/DeepSeek-V3"   # user-facing label
    api_model: str = _DEEPSEEK_CHAT_MODEL          # actual API model id
    token: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    cache_dir: str | Path = "artifacts/deepseek_cache"
    base_url: str = _DEEPSEEK_BASE_URL

    def __post_init__(self) -> None:
        self.token = (
            self.token
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("DEEPSEEK_TOKEN")
        )
        if not self.token:
            raise RuntimeError(
                "DeepSeekBackend requires DEEPSEEK_API_KEY in the environment "
                "(set via Colab Secrets). No token found."
            )
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required for DeepSeekBackend "
                "(`pip install openai`)."
            ) from exc
        # DeepSeek is OpenAI-compatible: same client, different base_url.
        self._client = OpenAI(api_key=self.token, base_url=self.base_url)

    @property
    def name(self) -> str:
        return f"deepseek:{self.model_name}"

    def _cache_key(self, prompt: str, *, max_tokens: int, temperature: float,
                   top_p: float, stop, seed: int) -> str:
        payload = {
            "api_model": self.api_model, "prompt": prompt,
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "stop": stop, "seed": seed,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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
        key = self._cache_key(prompt, max_tokens=max_tokens, temperature=temperature,
                              top_p=top_p, stop=stop, seed=seed)
        path = Path(self.cache_dir) / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return GenerationOutput(**payload)

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                seed=seed,
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
            finish = resp.choices[0].finish_reason or "stop"
            req_id = getattr(resp, "id", "")
        except Exception as exc:
            # Auth/quota failures must stop immediately with a clear message.
            raise RuntimeError(
                f"DeepSeek API call failed: {type(exc).__name__}: {str(exc)[:200]}"
            ) from exc

        out = GenerationOutput(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            finish=finish,
            backend=self.name,
            meta={"api_model": self.api_model, "request_id": req_id, "seed": seed},
        )
        # Persist for replay determinism.
        path.write_text(json.dumps(out.__dict__, ensure_ascii=False), encoding="utf-8")
        return out

    def count_tokens(self, text: str) -> int:
        # DeepSeek uses a cl100k-like tokenizer; tiktoken is a close proxy for
        # accounting purposes (exact server-side count returned in usage above).
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)
