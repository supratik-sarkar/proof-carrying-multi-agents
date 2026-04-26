"""
Multi-provider LLM client for the PCG-MAS demo.

The demo runs *live* by default but the Space owner pays nothing —
the default backend is the free Hugging Face Inference tier (rate-
limited, but plenty for reviewer traffic). Users who want to invoke
premium models (DeepSeek-V3, Llama-3.3-70B at scale, GPT-4, Claude,
etc.) bring their own API key, which is stored only in their browser
session and is never logged or persisted.

Supported providers (provider_id → human-friendly name):
    "hf_free"       Hugging Face Inference — free tier (default)
    "hf_byok"       Hugging Face Inference — user's own HF token
    "openai"        OpenAI / GPT-4 / GPT-4o
    "anthropic"     Anthropic / Claude
    "deepseek"      DeepSeek-V3 / DeepSeek-R1
    "together"      Together AI (Mixtral, Llama, Qwen)
    "groq"          Groq (extremely fast inference)
    "openrouter"    OpenRouter (gateway to many providers)

All providers expose the same `chat(messages, **kwargs) -> str`
interface, so the rest of the app never branches on provider.

Anonymity note: the client does NOT include any user-agent string,
referer, or fingerprinting header that could identify the Space owner.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# -----------------------------------------------------------------------------
# Provider catalog
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderInfo:
    id: str
    label: str
    requires_key: bool
    free: bool
    default_model: str
    available_models: tuple[str, ...]
    docs_url: str


PROVIDERS: dict[str, ProviderInfo] = {
    "hf_free": ProviderInfo(
        id="hf_free",
        label="HF Inference (free tier)",
        requires_key=False,
        free=True,
        default_model="meta-llama/Llama-3.2-3B-Instruct",
        available_models=(
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ),
        docs_url="https://huggingface.co/docs/api-inference",
    ),
    "hf_byok": ProviderInfo(
        id="hf_byok",
        label="HF Inference (your token)",
        requires_key=True,
        free=False,
        default_model="meta-llama/Llama-3.3-70B-Instruct",
        available_models=(
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "deepseek-ai/DeepSeek-V3",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ),
        docs_url="https://huggingface.co/settings/tokens",
    ),
    "openai": ProviderInfo(
        id="openai",
        label="OpenAI",
        requires_key=True,
        free=False,
        default_model="gpt-4o-mini",
        available_models=("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"),
        docs_url="https://platform.openai.com/api-keys",
    ),
    "anthropic": ProviderInfo(
        id="anthropic",
        label="Anthropic",
        requires_key=True,
        free=False,
        default_model="claude-3-5-sonnet-latest",
        available_models=(
            "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
        ),
        docs_url="https://console.anthropic.com/settings/keys",
    ),
    "deepseek": ProviderInfo(
        id="deepseek",
        label="DeepSeek",
        requires_key=True,
        free=False,
        default_model="deepseek-chat",
        available_models=("deepseek-chat", "deepseek-reasoner"),
        docs_url="https://platform.deepseek.com/api_keys",
    ),
    "together": ProviderInfo(
        id="together",
        label="Together AI",
        requires_key=True,
        free=False,
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        available_models=(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ),
        docs_url="https://api.together.ai/settings/api-keys",
    ),
    "groq": ProviderInfo(
        id="groq",
        label="Groq",
        requires_key=True,
        free=False,
        default_model="llama-3.3-70b-versatile",
        available_models=(
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
            "mixtral-8x7b-32768", "deepseek-r1-distill-llama-70b",
        ),
        docs_url="https://console.groq.com/keys",
    ),
    "openrouter": ProviderInfo(
        id="openrouter",
        label="OpenRouter",
        requires_key=True,
        free=False,
        default_model="meta-llama/llama-3.3-70b-instruct",
        available_models=(
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
        ),
        docs_url="https://openrouter.ai/keys",
    ),
}


# -----------------------------------------------------------------------------
# Provider-specific transports
# -----------------------------------------------------------------------------

def _http_post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float = 60.0,
) -> dict:
    """Stdlib HTTP POST (no requests dep). Raises LLMError on failure."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        raise LLMError(
            f"{e.code} {e.reason} from {url}: {err_body[:300]}"
        ) from e
    except Exception as e:
        raise LLMError(f"Request to {url} failed: {e}") from e


class LLMError(RuntimeError):
    """Raised on any provider call failure. Always safe to surface to the user
    (no API keys leaked)."""


# -----------------------------------------------------------------------------
# The unified client
# -----------------------------------------------------------------------------

class LLMClient:
    """One client, many providers. Constructed per-request from session state.

    Args:
        provider:  one of PROVIDERS' keys
        api_key:   user-supplied (BYOK) OR None for the default free tier
        model:     model id (must be a key in `PROVIDERS[provider].available_models`,
                   but we don't enforce because users may want bleeding-edge models
                   not yet listed)
    """

    def __init__(
        self,
        provider: str,
        api_key: str | None = None,
        model: str | None = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider!r}")
        info = PROVIDERS[provider]
        if info.requires_key and not api_key:
            raise LLMError(
                f"Provider {info.label!r} requires an API key. Paste yours in "
                "the sidebar (it will only be stored in your browser session)."
            )
        # For hf_free, use Space-side HF_TOKEN if set (gives rate-limit boost
        # but is OPTIONAL — works without one too). The token is server-side,
        # never exposed to clients.
        if provider == "hf_free" and not api_key:
            api_key = os.environ.get("HF_TOKEN") or None

        self.provider = provider
        self.info = info
        self.api_key = api_key
        self.model = model or info.default_model

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """Send a chat-completion request. Returns the assistant's text only.

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        """
        dispatch = {
            "hf_free":   self._call_hf,
            "hf_byok":   self._call_hf,
            "openai":    self._call_openai_compatible,
            "anthropic": self._call_anthropic,
            "deepseek":  self._call_deepseek,
            "together":  self._call_openai_compatible,
            "groq":      self._call_openai_compatible,
            "openrouter": self._call_openai_compatible,
        }
        return dispatch[self.provider](messages, temperature, max_tokens, kwargs)

    # ------------------------------------------------------------------
    # Provider-specific paths
    # ------------------------------------------------------------------

    def _call_hf(self, messages, temperature, max_tokens, _extra):
        """Hugging Face Inference API (router endpoint, OpenAI-compatible)."""
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = _http_post_json(url, payload, headers)
        return self._extract_openai_choice(data)

    def _call_openai_compatible(self, messages, temperature, max_tokens, _extra):
        """OpenAI / Together / Groq / OpenRouter all share the OpenAI schema."""
        urls = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "together": "https://api.together.xyz/v1/chat/completions",
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        }
        url = urls[self.provider]
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = _http_post_json(url, payload, headers)
        return self._extract_openai_choice(data)

    def _call_anthropic(self, messages, temperature, max_tokens, _extra):
        """Anthropic uses a slightly different schema (system separate from messages)."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        sys_messages = [m["content"] for m in messages if m.get("role") == "system"]
        chat_messages = [m for m in messages if m.get("role") != "system"]
        payload = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if sys_messages:
            payload["system"] = "\n\n".join(sys_messages)
        data = _http_post_json(url, payload, headers)
        # Anthropic shape: {"content": [{"type": "text", "text": "..."}], ...}
        try:
            blocks = data["content"]
            return "".join(b["text"] for b in blocks if b.get("type") == "text")
        except (KeyError, TypeError, IndexError) as e:
            raise LLMError(f"Anthropic response parse failed: {data}") from e

    def _call_deepseek(self, messages, temperature, max_tokens, _extra):
        """DeepSeek uses an OpenAI-compatible endpoint at platform.deepseek.com."""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = _http_post_json(url, payload, headers)
        return self._extract_openai_choice(data)

    @staticmethod
    def _extract_openai_choice(data: dict) -> str:
        """Extract assistant content from an OpenAI-shaped response."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, TypeError, IndexError) as e:
            raise LLMError(f"Response parse failed: {data}") from e


# -----------------------------------------------------------------------------
# Helpers used by the UI
# -----------------------------------------------------------------------------

def free_provider_info() -> ProviderInfo:
    """The default free provider. Always works (even without HF_TOKEN)."""
    return PROVIDERS["hf_free"]


def is_premium(provider_id: str) -> bool:
    return PROVIDERS.get(provider_id, free_provider_info()).requires_key


def list_premium_providers() -> list[ProviderInfo]:
    return [p for p in PROVIDERS.values() if p.requires_key]
