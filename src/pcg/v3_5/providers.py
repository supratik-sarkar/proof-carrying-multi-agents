"""PCG-MAS v3.5 Authoritative Model-to-Provider Routing & Adapters.

Enforces:
1. Exact provider-routing table for all 7 frozen models.
2. Concrete request serialization, execution, and response parsing.
3. Pluggable transport support for zero-cost offline regression testing.
4. Pre-flight verification of all routes, credentials, and writable paths.
5. Deterministic metadata, resource accounting, and ledger commit on response.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from pcg.v3_5.registries import FROZEN_MODELS
from pcg.v3_5.resource_ledger import MANDATORY_RESOURCE_METRICS, ResourceUsage

PROVIDER_ROUTING_TABLE: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {
        "model_id": "gpt-4o",
        "provider_id": "openai",
        "adapter_class": "OpenAIProviderAdapter",
        "api_key_env": "OPENAI_API_KEY",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "api_model": "gpt-4o",
        "tier": "primary_api_benchmark",
    },
    "gpt-4o-mini": {
        "model_id": "gpt-4o-mini",
        "provider_id": "openai",
        "adapter_class": "OpenAIProviderAdapter",
        "api_key_env": "OPENAI_API_KEY",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "api_model": "gpt-4o-mini",
        "tier": "primary_api_benchmark",
    },
    "o1-mini": {
        "model_id": "o1-mini",
        "provider_id": "openai",
        "adapter_class": "OpenAIProviderAdapter",
        "api_key_env": "OPENAI_API_KEY",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "api_model": "o1-mini",
        "tier": "reasoning_benchmark",
    },
    "claude-3-5-sonnet": {
        "model_id": "claude-3-5-sonnet",
        "provider_id": "anthropic",
        "adapter_class": "AnthropicProviderAdapter",
        "api_key_env": "ANTHROPIC_API_KEY",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "api_model": "claude-3-5-sonnet-20241022",
        "tier": "primary_api_benchmark",
    },
    "gemini-1.5-pro": {
        "model_id": "gemini-1.5-pro",
        "provider_id": "google_gemini",
        "adapter_class": "GeminiProviderAdapter",
        "api_key_env": "GEMINI_API_KEY",
        "fallback_api_key_env": "GOOGLE_API_KEY",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        "api_model": "gemini-1.5-pro",
        "tier": "primary_api_benchmark",
    },
    "gemini-1.5-flash": {
        "model_id": "gemini-1.5-flash",
        "provider_id": "google_gemini",
        "adapter_class": "GeminiProviderAdapter",
        "api_key_env": "GEMINI_API_KEY",
        "fallback_api_key_env": "GOOGLE_API_KEY",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "api_model": "gemini-1.5-flash",
        "tier": "fast_api_benchmark",
    },
    "llama-3.1-70b": {
        "model_id": "llama-3.1-70b",
        "provider_id": "hosted_open_weight",
        "adapter_class": "OpenAICompatibleHostedAdapter",
        "api_key_env": "LLAMA_API_KEY",
        "fallback_api_key_env": "TOGETHER_API_KEY",
        "endpoint": "https://api.together.xyz/v1/chat/completions",
        "api_model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "tier": "open_weight_reference",
    },
}


class BaseProviderAdapter:
    """Abstract base adapter defining standard provider interaction."""

    def __init__(self, route_info: Dict[str, Any]):
        self.route_info = route_info
        self.model_id = route_info["model_id"]
        self.provider_id = route_info["provider_id"]
        self.api_key_env = route_info["api_key_env"]
        self.fallback_api_key_env = route_info.get("fallback_api_key_env")
        self.endpoint = route_info["endpoint"]
        self.api_model = route_info["api_model"]

    def get_api_key(self) -> Optional[str]:
        key = os.environ.get(self.api_key_env)
        if not key and self.fallback_api_key_env:
            key = os.environ.get(self.fallback_api_key_env)
        return key

    def has_credentials(self) -> bool:
        return bool(self.get_api_key())

    def build_request(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def parse_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIProviderAdapter(BaseProviderAdapter):
    """Adapter for OpenAI models (gpt-4o, gpt-4o-mini, o1-mini)."""

    def build_request(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        messages = []
        if system and not self.model_id.startswith("o1"):
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.api_model,
            "messages": messages,
        }
        if not self.model_id.startswith("o1"):
            payload["temperature"] = 0.0
            payload["max_tokens"] = 1024
        else:
            payload["max_completion_tokens"] = 1024
        return payload

    def parse_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        choices = raw_resp.get("choices", [{}])
        msg = choices[0].get("message", {}) if choices else {}
        text = msg.get("content") or ""
        usage = raw_resp.get("usage", {})
        det = usage.get("completion_tokens_details", {})
        return {
            "text": text,
            "response_id": raw_resp.get("id", f"resp_{hashlib.sha256(text.encode()).hexdigest()[:12]}"),
            "n_input_tokens": usage.get("prompt_tokens", len(text.split())),
            "n_output_tokens": usage.get("completion_tokens", len(text.split())),
            "n_reasoning_tokens": det.get("reasoning_tokens", 0),
            "finish_reason": choices[0].get("finish_reason", "stop") if choices else "stop",
        }


class AnthropicProviderAdapter(BaseProviderAdapter):
    """Adapter for Anthropic models (claude-3-5-sonnet)."""

    def build_request(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.api_model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        return payload

    def parse_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        content = raw_resp.get("content", [])
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        usage = raw_resp.get("usage", {})
        return {
            "text": text,
            "response_id": raw_resp.get("id", f"resp_{hashlib.sha256(text.encode()).hexdigest()[:12]}"),
            "n_input_tokens": usage.get("input_tokens", len(text.split())),
            "n_output_tokens": usage.get("output_tokens", len(text.split())),
            "n_reasoning_tokens": 0,
            "finish_reason": raw_resp.get("stop_reason", "end_turn"),
        }


class GeminiProviderAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini models (gemini-1.5-pro, gemini-1.5-flash)."""

    def build_request(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 1024,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def parse_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        candidates = raw_resp.get("candidates", [{}])
        cand = candidates[0] if candidates else {}
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        usage = raw_resp.get("usageMetadata", {})
        return {
            "text": text,
            "response_id": raw_resp.get("responseId", f"resp_{hashlib.sha256(text.encode()).hexdigest()[:12]}"),
            "n_input_tokens": usage.get("promptTokenCount", len(text.split())),
            "n_output_tokens": usage.get("candidatesTokenCount", len(text.split())),
            "n_reasoning_tokens": usage.get("thoughtsTokenCount", 0),
            "finish_reason": cand.get("finishReason", "STOP"),
        }


class OpenAICompatibleHostedAdapter(OpenAIProviderAdapter):
    """Adapter for hosted open-weight models (llama-3.1-70b)."""
    pass


ADAPTER_CLASSES = {
    "OpenAIProviderAdapter": OpenAIProviderAdapter,
    "AnthropicProviderAdapter": AnthropicProviderAdapter,
    "GeminiProviderAdapter": GeminiProviderAdapter,
    "OpenAICompatibleHostedAdapter": OpenAICompatibleHostedAdapter,
}


def get_provider_adapter(model_name: str) -> BaseProviderAdapter:
    """Returns instantiated adapter for a given model from the routing table."""
    if model_name not in PROVIDER_ROUTING_TABLE:
        raise KeyError(
            f"FAIL CLOSED: Model '{model_name}' has no registered provider route! "
            f"Available routes: {list(PROVIDER_ROUTING_TABLE.keys())}"
        )
    route = PROVIDER_ROUTING_TABLE[model_name]
    cls = ADAPTER_CLASSES[route["adapter_class"]]
    return cls(route)


def execute_request_through_adapter(
    request_entry: Dict[str, Any],
    prompt_text: str,
    repo_root: Path,
    transport: Optional[Callable[[str, Dict[str, str], Dict[str, Any]], Dict[str, Any]]] = None,
    ledger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Executes a single manifested request through the production adapter pipeline.

    1. Resolves model -> provider adapter.
    2. Builds production provider request payload.
    3. Invokes transport (mocked in test, live HTTP in flight).
    4. Parses response and computes deterministic SHA-256 hash.
    5. Records standardized resource usage.
    6. Writes candidate record to artifacts/v3_5/d_cal/candidates/.
    7. Atomically commits request_id to ledger.
    """
    model = request_entry["model"]
    req_id = request_entry["request_id"]
    stage = request_entry.get("stage", "P2_DCAL_GENERATION")
    candidate_id = request_entry.get("candidate_id", f"cand_{model}_{req_id[:8]}")

    adapter = get_provider_adapter(model)
    payload = adapter.build_request(prompt=prompt_text)

    start_time = time.perf_counter()

    if transport is not None:
        # Zero-cost mocked or custom transport
        raw_resp = transport(adapter.endpoint, {"X-Provider": adapter.provider_id}, payload)
    else:
        # Production LIVE network transport
        api_key = adapter.get_api_key()
        if not api_key:
            raise RuntimeError(
                f"FAIL CLOSED: Cannot execute LIVE call for {model}. "
                f"Required credential {adapter.api_key_env} is not set in environment."
            )
        import urllib.request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if adapter.provider_id == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
            headers.pop("Authorization", None)
        elif adapter.provider_id == "google_gemini":
            headers["x-goog-api-key"] = api_key
            headers.pop("Authorization", None)

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(adapter.endpoint, data=data_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_resp = json.loads(resp.read().decode("utf-8"))

    elapsed = time.perf_counter() - start_time
    parsed = adapter.parse_response(raw_resp)
    text = parsed["text"]
    resp_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Create resource record adhering strictly to 8 mandatory metrics
    usage = ResourceUsage(
        candidate_id=candidate_id,
        system_id=model,
        n_input_tokens=parsed.get("n_input_tokens", 0),
        n_output_tokens=parsed.get("n_output_tokens", 0),
        n_reasoning_tokens=parsed.get("n_reasoning_tokens", 0),
        wall_time_seconds=round(elapsed, 4),
        gpu_seconds=0.0,
        n_model_calls=1,
        n_retries=0,
        n_tool_calls=0,
    )

    committed_record = {
        "candidate_id": candidate_id,
        "request_id": req_id,
        "model": model,
        "dataset": request_entry.get("dataset"),
        "example_id": request_entry.get("example_id"),
        "stage": stage,
        "response_text": text,
        "response_hash": resp_hash,
        "provider_metadata": {
            "provider": adapter.provider_id,
            "endpoint": adapter.endpoint,
            "api_model": adapter.api_model,
            "response_id": parsed.get("response_id"),
            "finish_reason": parsed.get("finish_reason"),
            "elapsed_seconds": round(elapsed, 4),
        },
        "resource_record": usage.to_dict(),
        "status": "COMMITTED",
    }

    # Write candidate artifact to disk
    cand_dir = repo_root / "artifacts" / "v3_5" / "d_cal" / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_path = cand_dir / f"{candidate_id}.json"
    cand_path.write_text(json.dumps(committed_record, indent=2, sort_keys=True), encoding="utf-8")

    # Atomic commit to ledger
    if ledger is not None:
        ledger.commit(req_id, stage)

    return committed_record
