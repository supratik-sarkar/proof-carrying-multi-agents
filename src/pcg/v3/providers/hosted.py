"""Hosted provider route (version-pinned). Credentials come from env at call time.

A key is never persisted, never logged, never placed in telemetry or a URL.
"""
from __future__ import annotations
import os, time
from typing import Any, Dict, Optional

from ..canon import sha256_text
from .base import ProviderResponse, make_fingerprint

REDACT = "***REDACTED***"


def redact(headers: Dict[str, str]) -> Dict[str, str]:
    return {k: (REDACT if k.lower() in ("authorization", "x-api-key", "api-key") else v)
            for k, v in headers.items()}


API_FIRST_SEVEN = {
    "openai": {"model_id": "gpt-5.6-sol", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"model_id": "claude-opus-5", "api_key_env": "ANTHROPIC_API_KEY"},
    "deepseek": {"model_id": "deepseek-v4-pro", "api_key_env": "DEEPSEEK_API_KEY"},
    "gemini": {"model_id": "gemini-3.1-pro-preview", "api_key_env": "GEMINI_API_KEY"},
    "xai": {"model_id": "grok-4.6", "api_key_env": "XAI_API_KEY"},
    "mistral": {"model_id": "mistral-medium-3-5", "api_key_env": "MISTRAL_API_KEY"},
    "cohere": {"model_id": "command-a-plus-05-2026", "api_key_env": "COHERE_API_KEY"},
}


class HostedProvider:
    route = "hosted_provider"

    def __init__(self, model_id: str, revision: str = "pinned",
                 api_key_env: str = "PCG_PROVIDER_API_KEY", base_url: Optional[str] = None,
                 pricing: Optional[Dict[str, float]] = None, seed: int = 0):
        self.model_id, self.revision, self.seed = model_id, revision, seed
        self.api_key_env, self.base_url = api_key_env, base_url
        self.pricing = pricing or {}

    def _key(self) -> str:
        k = os.environ.get(self.api_key_env)
        if not k:
            raise RuntimeError(f"{self.api_key_env} not set; keys are never stored in the repo")
        return k

    def generate(self, prompt: str, max_tokens: int = 256, **kw) -> ProviderResponse:
        raise RuntimeError(
            "HostedProvider.generate is intentionally unimplemented in the v3.0 remediation "
            "release: NETWORK_API_MODEL_CALLS must remain 0. Wire the client in the execution "
            "phase, where the pricing manifest and billed cost are recorded per call.")

    def cost(self, tokens_in: int, tokens_out: int) -> Optional[float]:
        pin, pout = self.pricing.get("input_per_1k"), self.pricing.get("output_per_1k")
        if pin is None or pout is None:
            return None
        return tokens_in / 1000 * pin + tokens_out / 1000 * pout


def get_api_provider(name: str, **kw) -> HostedProvider:
    if name not in API_FIRST_SEVEN:
        raise ValueError(f"Unknown API provider {name!r}. Available: {list(API_FIRST_SEVEN.keys())}")
    cfg = API_FIRST_SEVEN[name]
    return HostedProvider(model_id=cfg["model_id"], api_key_env=cfg["api_key_env"], **kw)
