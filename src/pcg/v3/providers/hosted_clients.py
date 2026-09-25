"""Seven hosted provider clients behind one contract.

Lazy: no SDK import, no client construction and no key read at import time. A
provider errors only when a real invocation is requested. Each client is
exercised offline against recorded transport fixtures.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..canon import sha256_text
from .contract import (ProviderCapabilities, ProviderError, ProviderRequest,
                       ProviderResponse, ProviderUsage, fingerprint_of, make_error)


class _Base:
    name = "base"
    key_env = "PCG_PROVIDER_API_KEY"
    default_caps = ProviderCapabilities()

    def capabilities(self) -> ProviderCapabilities:
        return self.default_caps

    def _key(self) -> str:
        k = os.environ.get(self.key_env)
        if not k:
            raise ProviderError("AUTH", 401, f"{self.key_env} not set", False)
        return k

    def invoke(self, req: ProviderRequest) -> ProviderResponse:
        raise ProviderError(
            "TRANSPORT", None,
            f"{self.name}.invoke is intentionally unimplemented in the architecture "
            "pass: PAID_MODEL_API_CALLS must remain 0.", False)

    # ------------------------------------------------------------- helpers
    def _resp(self, req, text, returned_model, revision, rid, usage, tools=None,
              finish=None) -> ProviderResponse:
        return ProviderResponse(
            text=text, requested_model=req.model, returned_model=returned_model,
            model_revision=revision, response_id=rid, provider=self.name,
            usage=usage, raw_output_sha256=sha256_text(text),
            tool_calls=tools or [], finish_reason=finish,
            backend_fingerprint=fingerprint_of(self.name, req, returned_model, revision))

    def map_error(self, status: int, payload: Dict[str, Any]) -> ProviderError:
        msg = (payload.get("error") or {}).get("message") if isinstance(
            payload.get("error"), dict) else payload.get("message") or "error"
        return make_error(status, str(msg))


class OpenAIClient(_Base):
    name = "openai"
    key_env = "OPENAI_API_KEY"
    default_caps = ProviderCapabilities(True, True, True, True, True, 128000,
                                        unsupported_params=[])

    def build_request(self, req: ProviderRequest) -> Dict[str, Any]:
        msgs = ([{"role": "system", "content": req.system}] if req.system else []) + \
               [{"role": "user", "content": req.prompt}]
        b: Dict[str, Any] = {"model": req.model, "messages": msgs,
                             "max_completion_tokens": req.max_tokens,
                             "temperature": req.temperature}
        if req.seed is not None:
            b["seed"] = req.seed
        if req.tools:
            b["tools"] = req.tools
        return b

    def parse_response(self, p: Dict[str, Any], req: ProviderRequest) -> ProviderResponse:
        ch = p["choices"][0]
        m = ch["message"]
        u = p.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        pdet = u.get("prompt_tokens_details") or {}
        return self._resp(
            req, m.get("content") or "", p.get("model"), p.get("system_fingerprint"),
            p.get("id"),
            ProviderUsage(u.get("prompt_tokens"), u.get("completion_tokens"),
                          det.get("reasoning_tokens"), pdet.get("cached_tokens")),
            [{"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
             for t in (m.get("tool_calls") or [])],
            ch.get("finish_reason"))


class AnthropicClient(_Base):
    name = "anthropic"
    key_env = "ANTHROPIC_API_KEY"
    default_caps = ProviderCapabilities(True, True, False, False, True, 200000,
                                        unsupported_params=["seed"])

    def build_request(self, req: ProviderRequest) -> Dict[str, Any]:
        b: Dict[str, Any] = {"model": req.model, "max_tokens": req.max_tokens,
                             "temperature": req.temperature,
                             "messages": [{"role": "user", "content": req.prompt}]}
        if req.system:
            b["system"] = req.system
        if req.tools:
            b["tools"] = req.tools
        return b                       # `seed` deliberately dropped: unsupported

    def parse_response(self, p: Dict[str, Any], req: ProviderRequest) -> ProviderResponse:
        text = "".join(c.get("text", "") for c in p.get("content", [])
                       if c.get("type") == "text")
        tools = [{"name": c.get("name"), "arguments": c.get("input")}
                 for c in p.get("content", []) if c.get("type") == "tool_use"]
        u = p.get("usage") or {}
        return self._resp(req, text, p.get("model"), None, p.get("id"),
                          ProviderUsage(u.get("input_tokens"), u.get("output_tokens"),
                                        None, u.get("cache_read_input_tokens")),
                          tools, p.get("stop_reason"))


class DeepSeekClient(OpenAIClient):
    name = "deepseek"
    key_env = "DEEPSEEK_API_KEY"
    default_caps = ProviderCapabilities(True, True, True, True, True, 64000)

    def parse_response(self, p: Dict[str, Any], req: ProviderRequest) -> ProviderResponse:
        r = super().parse_response(p, req)
        u = p.get("usage") or {}
        if u.get("prompt_cache_hit_tokens") is not None:
            r.usage.cached_tokens = u["prompt_cache_hit_tokens"]
        ch = p["choices"][0]["message"]
        if ch.get("reasoning_content"):
            r.usage.reasoning_tokens = u.get("completion_tokens_details", {}).get(
                "reasoning_tokens") or r.usage.reasoning_tokens
        return r


class GeminiClient(_Base):
    name = "google_gemini"
    key_env = "GOOGLE_API_KEY"
    default_caps = ProviderCapabilities(True, True, True, True, True, 1000000)

    def build_request(self, req: ProviderRequest) -> Dict[str, Any]:
        b: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
            "generationConfig": {"maxOutputTokens": req.max_tokens,
                                 "temperature": req.temperature}}
        if req.system:
            b["systemInstruction"] = {"parts": [{"text": req.system}]}
        if req.seed is not None:
            b["generationConfig"]["seed"] = req.seed
        if req.tools:
            b["tools"] = [{"functionDeclarations": req.tools}]
        return b

    def parse_response(self, p: Dict[str, Any], req: ProviderRequest) -> ProviderResponse:
        cand = (p.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(x.get("text", "") for x in parts if "text" in x)
        tools = [{"name": x["functionCall"]["name"],
                  "arguments": x["functionCall"].get("args")}
                 for x in parts if "functionCall" in x]
        u = p.get("usageMetadata") or {}
        return self._resp(req, text, p.get("modelVersion"), None, p.get("responseId"),
                          ProviderUsage(u.get("promptTokenCount"),
                                        u.get("candidatesTokenCount"),
                                        u.get("thoughtsTokenCount"),
                                        u.get("cachedContentTokenCount")),
                          tools, cand.get("finishReason"))


class XAIClient(OpenAIClient):
    name = "xai"
    key_env = "XAI_API_KEY"
    default_caps = ProviderCapabilities(True, True, True, True, False, 131072)


class MistralClient(OpenAIClient):
    name = "mistral"
    key_env = "MISTRAL_API_KEY"
    default_caps = ProviderCapabilities(True, True, True, False, False, 128000,
                                        unsupported_params=["reasoning"])

    def build_request(self, req: ProviderRequest) -> Dict[str, Any]:
        b = super().build_request(req)
        b["max_tokens"] = b.pop("max_completion_tokens")     # Mistral naming
        if req.seed is not None:
            b["random_seed"] = b.pop("seed")
        return b


class CohereClient(_Base):
    name = "cohere"
    key_env = "COHERE_API_KEY"
    default_caps = ProviderCapabilities(True, True, True, False, False, 128000)

    def build_request(self, req: ProviderRequest) -> Dict[str, Any]:
        msgs = ([{"role": "system", "content": req.system}] if req.system else []) + \
               [{"role": "user", "content": req.prompt}]
        b: Dict[str, Any] = {"model": req.model, "messages": msgs,
                             "max_tokens": req.max_tokens,
                             "temperature": req.temperature}
        if req.seed is not None:
            b["seed"] = req.seed
        if req.tools:
            b["tools"] = req.tools
        return b

    def parse_response(self, p: Dict[str, Any], req: ProviderRequest) -> ProviderResponse:
        msg = p.get("message") or {}
        text = "".join(c.get("text", "") for c in (msg.get("content") or []))
        tools = [{"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
                 for t in (msg.get("tool_calls") or [])]
        u = (p.get("usage") or {}).get("tokens") or {}
        return self._resp(req, text, p.get("model"), None, p.get("id"),
                          ProviderUsage(u.get("input_tokens"), u.get("output_tokens")),
                          tools, p.get("finish_reason"))


CLIENTS = {c.name: c for c in [OpenAIClient(), AnthropicClient(), DeepSeekClient(),
                               GeminiClient(), XAIClient(), MistralClient(),
                               CohereClient()]}
HOSTED_PROVIDERS = tuple(CLIENTS)
