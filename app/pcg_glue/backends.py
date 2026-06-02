"""BYOK backend resolver. Tokens live in user session only; never logged."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackendChoice:
    name: str            # display name
    provider: str        # 'openai' | 'anthropic' | 'deepseek' | 'hf_inference'
    model_id: str        # provider-specific model id
    key_env: str         # env var to set (informational only — we never persist)


# Curated short list — every entry uses a public, OpenAI-compatible or
# straightforward chat-completion endpoint.
PRESETS: list[BackendChoice] = [
    # OpenAI — frontier + flagship + cost-efficient
    BackendChoice("OpenAI · gpt-4.1",              "openai",       "gpt-4.1",                              "OPENAI_API_KEY"),
    BackendChoice("OpenAI · gpt-4.1-mini",         "openai",       "gpt-4.1-mini",                         "OPENAI_API_KEY"),
    BackendChoice("OpenAI · gpt-4o",               "openai",       "gpt-4o",                               "OPENAI_API_KEY"),
    BackendChoice("OpenAI · gpt-4o-mini",          "openai",       "gpt-4o-mini",                          "OPENAI_API_KEY"),
    BackendChoice("OpenAI · o4-mini (reasoning)",  "openai",       "o4-mini",                              "OPENAI_API_KEY"),
    # Anthropic — current flagship + cost-efficient
    BackendChoice("Anthropic · claude-sonnet-4.5", "anthropic",    "claude-sonnet-4-5",                    "ANTHROPIC_API_KEY"),
    BackendChoice("Anthropic · claude-opus-4.1",   "anthropic",    "claude-opus-4-1",                      "ANTHROPIC_API_KEY"),
    BackendChoice("Anthropic · claude-haiku-4.5",  "anthropic",    "claude-haiku-4-5",                     "ANTHROPIC_API_KEY"),
    BackendChoice("Anthropic · claude-3-5-sonnet", "anthropic",    "claude-3-5-sonnet-latest",             "ANTHROPIC_API_KEY"),
    # DeepSeek — V3 via official API
    BackendChoice("DeepSeek · deepseek-chat (V3)", "deepseek",     "deepseek-chat",                        "DEEPSEEK_API_KEY"),
    BackendChoice("DeepSeek · deepseek-reasoner",  "deepseek",     "deepseek-reasoner",                    "DEEPSEEK_API_KEY"),
    # HF Inference — open-weight, larger models
    BackendChoice("HF Inference · Llama-3.3-70B",  "hf_inference", "meta-llama/Llama-3.3-70B-Instruct",    "HF_TOKEN"),
    BackendChoice("HF Inference · Llama-3.1-70B",  "hf_inference", "meta-llama/Llama-3.1-70B-Instruct",    "HF_TOKEN"),
    BackendChoice("HF Inference · Llama-3.1-8B",   "hf_inference", "meta-llama/Llama-3.1-8B-Instruct",     "HF_TOKEN"),
    BackendChoice("HF Inference · Mixtral-8x7B",   "hf_inference", "mistralai/Mixtral-8x7B-Instruct-v0.1", "HF_TOKEN"),
    BackendChoice("HF Inference · Qwen-2.5-72B",   "hf_inference", "Qwen/Qwen2.5-72B-Instruct",            "HF_TOKEN"),
]

PROVIDER_HINTS = {
    "openai":       "Get a key at platform.openai.com/api-keys (starts with sk-…)",
    "anthropic":    "Get a key at console.anthropic.com/settings/keys (starts with sk-ant-…)",
    "deepseek":     "Get a key at platform.deepseek.com (starts with sk-…)",
    "hf_inference": "Use a HF read token from huggingface.co/settings/tokens (starts with hf_…)",
}


def choice_by_label(label: str) -> Optional[BackendChoice]:
    return next((c for c in PRESETS if c.name == label), None)


def call_chat(choice: BackendChoice, api_key: str, prompt: str, *,
              system: str | None = None, max_tokens: int = 512,
              temperature: float = 0.0) -> tuple[str, dict]:
    """Provider-agnostic chat call. Returns (text, meta).

    No persistence: the api_key is used inside this call only and never logged.
    """
    if not api_key:
        raise RuntimeError(f"No API key provided for {choice.provider}. {PROVIDER_HINTS.get(choice.provider, '')}")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if choice.provider in ("openai", "deepseek"):
        from openai import OpenAI
        base_url = "https://api.deepseek.com" if choice.provider == "deepseek" else None
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=choice.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        meta = {
            "provider": choice.provider,
            "model": choice.model_id,
            "tokens_in":  getattr(usage, "prompt_tokens", 0) if usage else 0,
            "tokens_out": getattr(usage, "completion_tokens", 0) if usage else 0,
            "finish": resp.choices[0].finish_reason,
        }
        return text, meta

    if choice.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {
            "model": choice.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        meta = {
            "provider": "anthropic",
            "model": choice.model_id,
            "tokens_in":  resp.usage.input_tokens,
            "tokens_out": resp.usage.output_tokens,
            "finish": resp.stop_reason,
        }
        return text, meta

    if choice.provider == "hf_inference":
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=choice.model_id, token=api_key)
        resp = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        meta = {
            "provider": "hf_inference",
            "model": choice.model_id,
            "tokens_in":  getattr(getattr(resp, "usage", None), "prompt_tokens", 0),
            "tokens_out": getattr(getattr(resp, "usage", None), "completion_tokens", 0),
            "finish": "stop",
        }
        return text, meta

    raise ValueError(f"Unknown provider: {choice.provider}")
