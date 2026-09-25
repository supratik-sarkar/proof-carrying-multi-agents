"""One typed provider contract for all backends.

No hosted provider is called in this pass. Each client is exercised end-to-end
against RECORDED transport fixtures, so `PROVIDERS_PASSING_RECORDED_CONFORMANCE`
means the client was actually run -- not merely that a file exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from ..canon import fingerprint, sha256_text
from ..exec.retry import RetryTriggerClass, classify

REDACTED = "***REDACTED***"
SECRET_HEADERS = ("authorization", "x-api-key", "api-key", "x-goog-api-key")


class ErrorClass(str, Enum):
    AUTH = "AUTH"
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    TRANSPORT = "TRANSPORT"
    BAD_REQUEST = "BAD_REQUEST"
    CONTENT_FILTER = "CONTENT_FILTER"
    UNKNOWN = "UNKNOWN"


@dataclass
class ProviderCapabilities:
    supports_tools: bool = False
    supports_system_prompt: bool = True
    supports_seed: bool = False
    supports_reasoning_tokens: bool = False
    supports_cached_tokens: bool = False
    max_output_tokens: Optional[int] = None
    unsupported_params: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ProviderRequest:
    model: str
    prompt: str
    system: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.0
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def decoding_config(self) -> Dict[str, Any]:
        return {"max_tokens": self.max_tokens, "temperature": self.temperature,
                "seed": self.seed}


@dataclass
class ProviderUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ProviderError(Exception):
    error_class: str
    status_code: Optional[int]
    message: str
    retryable: bool
    retry_trigger_class: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.error_class}({self.status_code}): {self.message}"


@dataclass
class ProviderRetryEvent:
    attempt: int
    status_code: Optional[int]
    trigger_class: str
    delay_s: float


@dataclass
class ProviderResponse:
    text: str
    requested_model: str
    returned_model: Optional[str]
    model_revision: Optional[str]
    response_id: Optional[str]
    provider: str
    usage: ProviderUsage
    raw_output_sha256: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0                 # observed
    billed_cost_usd: Optional[float] = None  # observed
    retries: List[ProviderRetryEvent] = field(default_factory=list)  # observed
    backend_fingerprint: str = ""

    @property
    def committed(self) -> Dict[str, Any]:
        """Address-forming fields only."""
        return {"selected_output_sha256": self.raw_output_sha256,
                "model_id": self.requested_model,
                "model_revision": self.model_revision,
                "returned_model": self.returned_model}

    def to_dict(self) -> dict:
        d = dict(vars(self))
        d["usage"] = self.usage.to_dict()
        d["retries"] = [vars(r) for r in self.retries]
        return d


def redact_headers(h: Dict[str, str]) -> Dict[str, str]:
    return {k: (REDACTED if k.lower() in SECRET_HEADERS else v) for k, v in h.items()}


class ProviderClient(Protocol):
    name: str
    def capabilities(self) -> ProviderCapabilities: ...
    def build_request(self, req: ProviderRequest) -> Dict[str, Any]: ...
    def parse_response(self, payload: Dict[str, Any], req: ProviderRequest) -> ProviderResponse: ...
    def map_error(self, status: int, payload: Dict[str, Any]) -> ProviderError: ...


def make_error(status: int, message: str) -> ProviderError:
    trig = classify(status_code=status)
    if status in (401, 403):
        cls = ErrorClass.AUTH
    elif status == 429:
        cls = ErrorClass.RATE_LIMIT
    elif status in (408, 504):
        cls = ErrorClass.TIMEOUT
    elif 500 <= status < 600:
        cls = ErrorClass.TRANSPORT
    elif status == 400:
        cls = ErrorClass.BAD_REQUEST
    else:
        cls = ErrorClass.UNKNOWN
    return ProviderError(cls.value, status, message, trig is not None,
                         trig.value if trig else None)


def fingerprint_of(provider: str, req: ProviderRequest, returned_model: Optional[str],
                   revision: Optional[str]) -> str:
    return fingerprint(provider=provider, requested_model=req.model,
                       returned_model=returned_model, revision=revision,
                       decoding=req.decoding_config, tools=bool(req.tools))
