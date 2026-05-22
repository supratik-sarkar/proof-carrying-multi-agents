"""
Remote Hugging Face inference backend.

Tokens are resolved through `pcg.utils.hf_auth` and are never read from
source-controlled files, printed, or written into result artifacts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcg.backends.base import GenerationOutput, LLMBackend
from pcg.utils.hf_auth import require_hf_token_for_remote_backend


@dataclass
class HFInferenceBackend(LLMBackend):
    model_name: str
    token: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    cache_dir: str | Path = "artifacts/hf_cache"

    def __post_init__(self) -> None:
        self.token = require_hf_token_for_remote_backend(self.token)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for HFInferenceBackend. "
                "Install it with `pip install huggingface_hub`."
            ) from exc

        self._client = InferenceClient(model=self.model_name, token=self.token)

    @property
    def name(self) -> str:
        return f"hf-inference:{self.model_name}"

    def _cache_key(self, prompt: str, seed: int | None = None, **kwargs: Any) -> str:
        payload = {
            "model_name": self.model_name,
            "prompt": prompt,
            "seed": seed,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "kwargs": kwargs,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return Path(self.cache_dir) / f"{key}.json"

    def generate(self, prompt: str, seed: int | None = None, **kwargs: Any) -> GenerationOutput:
        key = self._cache_key(prompt, seed=seed, **kwargs)
        path = self._cache_path(key)

        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return GenerationOutput(**payload)

        # Prefer chat_completion when available; fall back to text_generation only
        # for genuine task/method compatibility issues. Auth/permission failures
        # must stop immediately, otherwise users see a misleading secondary error.
        try:
            response = self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                seed=seed,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            billing_or_quota = (
                "402" in msg
                or "payment required" in low
                or "depleted the monthly included credits" in low
                or "pre-paid credits" in low
                or "included credits" in low
            )
            if billing_or_quota:
                raise RuntimeError(
                    "HF Inference Providers billing/quota failed. Your HF token works, "
                    "but the account has depleted its included hosted-inference credits "
                    "or requires billing/pre-paid credits. Either add HF credits / use PRO, "
                    "switch to --backend hf_local on Colab/Databricks, or use --backend mock "
                    "for artifact/preflight runs."
                ) from exc

            auth_or_permission = (
                "403" in msg
                or "401" in msg
                or "forbidden" in low
                or "unauthorized" in low
                or "permission" in low
                or "authentication method" in low
                or "not allowed to call" in low
            )
            if auth_or_permission:
                raise RuntimeError(
                    "HF Inference Provider permission failed. Your HF token was found, "
                    "but it is not allowed to call Hugging Face Inference Providers "
                    "through router.huggingface.co for this model. Create/use a token "
                    "with Inference Providers permission, or run with --backend hf_local "
                    "on Colab/Databricks, or use --backend mock for preflight."
                ) from exc

            try:
                text = self._client.text_generation(
                    prompt,
                    max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    seed=seed,
                )
            except Exception as text_exc:
                raise RuntimeError(
                    f"HF inference failed for model {self.model_name}. "
                    "chat_completion failed and text_generation fallback also failed. "
                    f"Original chat error: {exc}. Text-generation error: {text_exc}"
                ) from text_exc

        out = GenerationOutput(
            text=str(text),
            tokens_in=len(prompt.split()),
            tokens_out=len(str(text).split()),
            backend=self.name,
            meta={
                "model_name": self.model_name,
                "cached": False,
            },
        )

        # Never write tokens/secrets to cache.
        path.write_text(json.dumps(out.__dict__, indent=2), encoding="utf-8")
        return out
