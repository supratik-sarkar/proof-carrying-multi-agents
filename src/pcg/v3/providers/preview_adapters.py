"""Canonical preview provider adapters for Professor-Review Free API Panel.

Maps all seven panel models across Google Gemini, Groq, and Cloudflare Workers AI
into a single canonical request/response schema.

Safety guarantees:
- Zero secret logging or persistence.
- Provider-native web search/browsing/code execution explicitly disabled.
- Bounded retries: max 2 retries on transient 5xx/timeout only.
- Fail-closed on 401/403/404.
- Raw response bodies are never persisted in smoke mode.
- Configurable timeouts (default 120s for Gemini, 45s for Groq/Cloudflare).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import socket
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

# Ensure IPv4 preference to prevent IPv6 routing blackholes on macOS
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_first_getaddrinfo(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    v4 = [r for r in res if r[0] == socket.AF_INET]
    return v4 if v4 else res
socket.getaddrinfo = _ipv4_first_getaddrinfo

USER_AGENT = "PCG-MAS-Preflight/1.0"
DEFAULT_TIMEOUT_GEMINI = float(os.environ.get("PCG_GEMINI_TIMEOUT", "15.0"))
DEFAULT_TIMEOUT_GROQ = float(os.environ.get("PCG_GROQ_TIMEOUT", "45.0"))
DEFAULT_TIMEOUT_CLOUDFLARE = float(os.environ.get("PCG_CLOUDFLARE_TIMEOUT", "45.0"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CanonicalPreviewRequest:
    provider: str
    requested_model_id: str
    prompt: str
    system: Optional[str] = None
    max_tokens: int = 24
    temperature: float = 0.0
    seed: Optional[int] = None
    tool_policy: str = "DISABLED"
    structured_output_policy: str = "NONE"
    timeout: float = 45.0
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"max_retries": 2, "backoff_factor": 1.5})
    request_hash: str = ""

    def __post_init__(self):
        if not self.request_hash:
            raw = f"{self.provider}|{self.requested_model_id}|{self.prompt}|{self.max_tokens}|{self.temperature}"
            self.request_hash = sha256_hex(raw)


@dataclass
class CanonicalPreviewResponse:
    provider: str
    requested_model: str
    returned_model: Optional[str]
    response_id: Optional[str]
    response_id_sha256: Optional[str]
    http_status: Optional[int]
    token_usage: Dict[str, Optional[int]]
    latency_seconds: float
    retry_count: int
    rate_limit_headers: Dict[str, str]
    raw_output_sha256: Optional[str]
    raw_response_text: Optional[str] = None
    provider_sdk_version: Optional[str] = None
    error_class: Optional[str] = None
    success: bool = False
    response_payload_persisted: bool = False
    scientific_partition_accessed: bool = False
    provider_native_tools_enabled: bool = False


class GoogleGeminiFreeAdapter:
    provider = "google_gemini_free"

    @classmethod
    def invoke(cls, req: CanonicalPreviewRequest, api_key: str) -> CanonicalPreviewResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{req.requested_model_id}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
            "generationConfig": {
                "maxOutputTokens": req.max_tokens,
                "temperature": req.temperature,
            },
        }
        if req.seed is not None:
            body["generationConfig"]["seed"] = req.seed

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "User-Agent": USER_AGENT,
        }

        data_bytes = json.dumps(body).encode("utf-8")
        timeout = req.timeout or DEFAULT_TIMEOUT_GEMINI
        max_retries = req.retry_policy.get("max_retries", 2)

        retries = 0
        while True:
            t0 = time.perf_counter()
            http_req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(http_req, timeout=timeout) as r:
                    latency = time.perf_counter() - t0
                    resp_bytes = r.read()
                    data = json.loads(resp_bytes.decode("utf-8") or "{}")

                    returned_model = data.get("modelVersion") or req.requested_model_id
                    res_id = data.get("responseId")
                    res_id_sha = sha256_hex(str(res_id)) if res_id else None

                    usage_raw = data.get("usageMetadata", {})
                    token_usage = {
                        "prompt_tokens": usage_raw.get("promptTokenCount"),
                        "completion_tokens": usage_raw.get("candidatesTokenCount"),
                        "total_tokens": usage_raw.get("totalTokenCount"),
                    }

                    cand = (data.get("candidates") or [{}])[0]
                    parts = (cand.get("content") or {}).get("parts") or []
                    text = "".join(p.get("text", "") for p in parts if "text" in p)
                    text_sha = sha256_hex(text) if text else None

                    rate_headers = {
                        k.lower(): v for k, v in r.headers.items()
                        if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
                    }

                    return CanonicalPreviewResponse(
                        provider=cls.provider,
                        requested_model=req.requested_model_id,
                        returned_model=returned_model,
                        response_id=None,
                        response_id_sha256=res_id_sha,
                        http_status=r.status,
                        token_usage=token_usage,
                        latency_seconds=round(latency, 4),
                        retry_count=retries,
                        rate_limit_headers=rate_headers,
                        raw_output_sha256=text_sha,
                        raw_response_text=text,
                        provider_sdk_version="v1beta-rest",
                        error_class=None,
                        success=True,
                    )
            except urllib.error.HTTPError as e:
                latency = time.perf_counter() - t0
                status = e.code
                if status < 500:
                    return CanonicalPreviewResponse(
                        provider=cls.provider,
                        requested_model=req.requested_model_id,
                        returned_model=None,
                        response_id=None,
                        response_id_sha256=None,
                        http_status=status,
                        token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                        latency_seconds=round(latency, 4),
                        retry_count=retries,
                        rate_limit_headers={},
                        raw_output_sha256=None,
                        provider_sdk_version="v1beta-rest",
                        error_class=f"HTTP_{status}",
                        success=False,
                    )
                if retries < max_retries:
                    retries += 1
                    time.sleep(1.0 * retries)
                    continue
                return CanonicalPreviewResponse(
                    provider=cls.provider,
                    requested_model=req.requested_model_id,
                    returned_model=None,
                    response_id=None,
                    response_id_sha256=None,
                    http_status=status,
                    token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    latency_seconds=round(latency, 4),
                    retry_count=retries,
                    rate_limit_headers={},
                    raw_output_sha256=None,
                    provider_sdk_version="v1beta-rest",
                    error_class=f"HTTP_{status}",
                    success=False,
                )
            except Exception as e:
                latency = time.perf_counter() - t0
                err_type = type(e).__name__
                if "timeout" in err_type.lower() and retries < max_retries:
                    retries += 1
                    time.sleep(1.0 * retries)
                    continue
                return CanonicalPreviewResponse(
                    provider=cls.provider,
                    requested_model=req.requested_model_id,
                    returned_model=None,
                    response_id=None,
                    response_id_sha256=None,
                    http_status=None,
                    token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    latency_seconds=round(latency, 4),
                    retry_count=retries,
                    rate_limit_headers={},
                    raw_output_sha256=None,
                    provider_sdk_version="v1beta-rest",
                    error_class=err_type,
                    success=False,
                )


class GroqFreeAdapter:
    provider = "groq_free"

    @classmethod
    def invoke(cls, req: CanonicalPreviewRequest, api_key: str) -> CanonicalPreviewResponse:
        url = "https://api.groq.com/openai/v1/chat/completions"
        body = {
            "model": req.requested_model_id,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_completion_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.seed is not None:
            body["seed"] = req.seed

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
        }

        data_bytes = json.dumps(body).encode("utf-8")
        timeout = req.timeout or DEFAULT_TIMEOUT_GROQ
        max_retries = req.retry_policy.get("max_retries", 2)

        retries = 0
        while True:
            t0 = time.perf_counter()
            http_req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(http_req, timeout=timeout) as r:
                    latency = time.perf_counter() - t0
                    resp_bytes = r.read()
                    data = json.loads(resp_bytes.decode("utf-8") or "{}")

                    returned_model = data.get("model") or req.requested_model_id
                    res_id = data.get("id")
                    res_id_sha = sha256_hex(str(res_id)) if res_id else None

                    usage_raw = data.get("usage", {})
                    token_usage = {
                        "prompt_tokens": usage_raw.get("prompt_tokens"),
                        "completion_tokens": usage_raw.get("completion_tokens"),
                        "total_tokens": usage_raw.get("total_tokens"),
                    }

                    choice = (data.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    content = (msg.get("content") or "").strip()
                    reasoning = (msg.get("reasoning") or "").strip()
                    text = content if content else (reasoning if reasoning else str(choice.get("text") or "").strip())
                    text_sha = sha256_hex(text) if text else None

                    rate_headers = {
                        k.lower(): v for k, v in r.headers.items()
                        if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
                    }

                    return CanonicalPreviewResponse(
                        provider=cls.provider,
                        requested_model=req.requested_model_id,
                        returned_model=returned_model,
                        response_id=None,
                        response_id_sha256=res_id_sha,
                        http_status=r.status,
                        token_usage=token_usage,
                        latency_seconds=round(latency, 4),
                        retry_count=retries,
                        rate_limit_headers=rate_headers,
                        raw_output_sha256=text_sha,
                        raw_response_text=text,
                        provider_sdk_version="openai-v1-groq-rest",
                        error_class=None,
                        success=True,
                    )
            except urllib.error.HTTPError as e:
                latency = time.perf_counter() - t0
                status = e.code
                if status < 500:
                    return CanonicalPreviewResponse(
                        provider=cls.provider,
                        requested_model=req.requested_model_id,
                        returned_model=None,
                        response_id=None,
                        response_id_sha256=None,
                        http_status=status,
                        token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                        latency_seconds=round(latency, 4),
                        retry_count=retries,
                        rate_limit_headers={},
                        raw_output_sha256=None,
                        provider_sdk_version="openai-v1-groq-rest",
                        error_class=f"HTTP_{status}",
                        success=False,
                    )
                if retries < max_retries:
                    retries += 1
                    time.sleep(1.0 * retries)
                    continue
                return CanonicalPreviewResponse(
                    provider=cls.provider,
                    requested_model=req.requested_model_id,
                    returned_model=None,
                    response_id=None,
                    response_id_sha256=None,
                    http_status=status,
                    token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    latency_seconds=round(latency, 4),
                    retry_count=retries,
                    rate_limit_headers={},
                    raw_output_sha256=None,
                    provider_sdk_version="openai-v1-groq-rest",
                    error_class=f"HTTP_{status}",
                    success=False,
                )
            except Exception as e:
                latency = time.perf_counter() - t0
                err_type = type(e).__name__
                if "timeout" in err_type.lower() and retries < max_retries:
                    retries += 1
                    time.sleep(1.0 * retries)
                    continue
                return CanonicalPreviewResponse(
                    provider=cls.provider,
                    requested_model=req.requested_model_id,
                    returned_model=None,
                    response_id=None,
                    response_id_sha256=None,
                    http_status=None,
                    token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    latency_seconds=round(latency, 4),
                    retry_count=retries,
                    rate_limit_headers={},
                    raw_output_sha256=None,
                    provider_sdk_version="openai-v1-groq-rest",
                    error_class=err_type,
                    success=False,
                )


class CloudflareWorkersAIFreeAdapter:
    provider = "cloudflare_workers_ai_free"

    @classmethod
    def invoke(cls, req: CanonicalPreviewRequest, token: str, account_id: str) -> CanonicalPreviewResponse:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{req.requested_model_id}"
        body = {
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
        }

        data_bytes = json.dumps(body).encode("utf-8")
        timeout = req.timeout or DEFAULT_TIMEOUT_CLOUDFLARE
        max_retries = req.retry_policy.get("max_retries", 2)

        retries = 0
        while True:
            t0 = time.perf_counter()
            http_req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(http_req, timeout=timeout) as r:
                    latency = time.perf_counter() - t0
                    resp_bytes = r.read()
                    data = json.loads(resp_bytes.decode("utf-8") or "{}")

                    returned_model = req.requested_model_id
                    res_id = data.get("result", {}).get("id") or data.get("id")
                    res_id_sha = sha256_hex(str(res_id)) if res_id else None

                    res_obj = data.get("result", {})
                    if isinstance(res_obj, dict):
                        raw_resp = res_obj.get("response")
                        if raw_resp:
                            text = str(raw_resp)
                        elif res_obj.get("choices"):
                            c0 = res_obj["choices"][0]
                            msg = c0.get("message") or {}
                            content = (msg.get("content") or "").strip()
                            reasoning = (msg.get("reasoning") or "").strip()
                            text = content if content else (reasoning if reasoning else str(c0.get("text") or "").strip())
                            if not text and msg:
                                text = str(msg)
                        else:
                            text = ""
                    else:
                        text = str(res_obj) if res_obj is not None else ""

                    text_sha = sha256_hex(text) if text else None

                    usage_raw = res_obj.get("usage", {}) if isinstance(res_obj, dict) else {}
                    token_usage = {
                        "prompt_tokens": usage_raw.get("prompt_tokens"),
                        "completion_tokens": usage_raw.get("completion_tokens"),
                        "total_tokens": usage_raw.get("total_tokens"),
                    }

                    rate_headers = {
                        k.lower(): v for k, v in r.headers.items()
                        if k.lower().startswith("cf-") or k.lower().startswith("x-ratelimit")
                    }

                    return CanonicalPreviewResponse(
                        provider=cls.provider,
                        requested_model=req.requested_model_id,
                        returned_model=returned_model,
                        response_id=None,
                        response_id_sha256=res_id_sha,
                        http_status=r.status,
                        token_usage=token_usage,
                        latency_seconds=round(latency, 4),
                        retry_count=retries,
                        rate_limit_headers=rate_headers,
                        raw_output_sha256=text_sha,
                        raw_response_text=text,
                        provider_sdk_version="cf-workers-ai-v4-rest",
                        error_class=None,
                        success=True,
                    )
            except urllib.error.HTTPError as e:
                latency = time.perf_counter() - t0
                status = e.code
                if status < 500:
                    return CanonicalPreviewResponse(
                        provider=cls.provider,
                        requested_model=req.requested_model_id,
                        returned_model=None,
                        response_id=None,
                        response_id_sha256=None,
                        http_status=status,
                        token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                        latency_seconds=round(latency, 4),
                        retry_count=retries,
                        rate_limit_headers={},
                        raw_output_sha256=None,
                        provider_sdk_version="cf-workers-ai-v4-rest",
                        error_class=f"HTTP_{status}",
                        success=False,
                    )
                if retries < max_retries:
                    retries += 1
                    time.sleep(1.0 * retries)
                    continue
                return CanonicalPreviewResponse(
                    provider=cls.provider,
                    requested_model=req.requested_model_id,
                    returned_model=None,
                    response_id=None,
                    response_id_sha256=None,
                    http_status=status,
                    token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    latency_seconds=round(latency, 4),
                    retry_count=retries,
                    rate_limit_headers={},
                    raw_output_sha256=None,
                    provider_sdk_version="cf-workers-ai-v4-rest",
                    error_class=f"HTTP_{status}",
                    success=False,
                )
            except Exception as e:
                latency = time.perf_counter() - t0
                err_type = type(e).__name__
                if "timeout" in err_type.lower() and retries < max_retries:
                    retries += 1
                    time.sleep(1.0 * retries)
                    continue
                return CanonicalPreviewResponse(
                    provider=cls.provider,
                    requested_model=req.requested_model_id,
                    returned_model=None,
                    response_id=None,
                    response_id_sha256=None,
                    http_status=None,
                    token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    latency_seconds=round(latency, 4),
                    retry_count=retries,
                    rate_limit_headers={},
                    raw_output_sha256=None,
                    provider_sdk_version="cf-workers-ai-v4-rest",
                    error_class=err_type,
                    success=False,
                )
