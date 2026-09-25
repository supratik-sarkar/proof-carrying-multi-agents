"""Transport-only retry classification (D4).

A retry may be triggered ONLY by a transport-level condition. Content-conditioned
regeneration is controller/repair behaviour with its own provenance, never a
transport retry: conditioning on "a later attempt succeeded" changes the
composition of the answered set and would move mass into selectivity while
appearing as verification.

If the frozen v3.2 controller specifies a repair action, route regeneration
through `controller_action`; do not assume it is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

MAX_ATTEMPTS = 3


class RetryTriggerClass(str, Enum):
    TRANSPORT = "TRANSPORT"
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"


ALLOWED_TRIGGERS = frozenset(t.value for t in RetryTriggerClass)

#: Explicitly forbidden triggers, named so the failure message is unambiguous.
FORBIDDEN_TRIGGERS = {
    "POOR_ANSWER", "CHECKER_FAILED", "PCG_REJECTED", "WEAK_EVIDENCE",
    "CONTROLLER_PREFERENCE", "LOW_CONFIDENCE", "ENTAILMENT_FAILED",
}


class ForbiddenRetryTrigger(ValueError):
    pass


def classify(status_code: Optional[int] = None, exception: Optional[BaseException] = None,
             ) -> Optional[RetryTriggerClass]:
    """Map a transport outcome to a retry class, or None for do-not-retry."""
    if status_code is not None:
        if status_code == 429:
            return RetryTriggerClass.RATE_LIMIT
        if status_code in (408, 504):
            return RetryTriggerClass.TIMEOUT
        if 500 <= status_code < 600:
            return RetryTriggerClass.TRANSPORT
        return None                       # 4xx other than the above: never retry
    if exception is not None:
        name = type(exception).__name__.lower()
        if "timeout" in name:
            return RetryTriggerClass.TIMEOUT
        if any(k in name for k in ("connection", "socket", "transport", "protocol")):
            return RetryTriggerClass.TRANSPORT
    return None


def assert_transport_only(trigger: str) -> RetryTriggerClass:
    up = str(trigger).upper()
    if up in FORBIDDEN_TRIGGERS or up not in ALLOWED_TRIGGERS:
        raise ForbiddenRetryTrigger(
            f"retry trigger {trigger!r} is not transport-level. Content-conditioned "
            "regeneration is a controller/repair action and must be recorded as "
            "controller_action, never as a transport retry.")
    return RetryTriggerClass(up)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = MAX_ATTEMPTS
    base_delay_s: float = 0.5
    multiplier: float = 2.0
    jitter_s: float = 0.25
    allowed: tuple = tuple(sorted(ALLOWED_TRIGGERS))

    def delay(self, attempt: int, rand: float = 0.5) -> float:
        """Bounded exponential backoff with jitter. Deterministic given `rand`."""
        return self.base_delay_s * (self.multiplier ** max(0, attempt - 1)) + self.jitter_s * rand

    @property
    def identity(self) -> Dict[str, Any]:
        """Enters spec_hash; must be identical across A05/A16 arms."""
        return {"max_attempts": self.max_attempts, "base_delay_s": self.base_delay_s,
                "multiplier": self.multiplier, "jitter_s": self.jitter_s,
                "allowed_triggers": list(self.allowed)}


@dataclass
class AttemptRecord:
    """One PHYSICAL attempt. Many attempts map to ONE logical inference."""
    attempt: int
    trigger: Optional[str]
    status_code: Optional[int]
    latency_ms: float
    span_id: Optional[str] = None
    succeeded: bool = False

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class LogicalInference:
    """The committed unit. Selected content is committed; attempt number is observed."""
    selected_output_sha256: str
    attempts: List[AttemptRecord] = field(default_factory=list)

    @property
    def physical_attempts(self) -> int:
        return len(self.attempts)

    @property
    def committed(self) -> Dict[str, Any]:
        return {"selected_output_sha256": self.selected_output_sha256}

    @property
    def observed(self) -> Dict[str, Any]:
        return {"retry_count": self.physical_attempts,
                "retry_timings_ms": [a.latency_ms for a in self.attempts]}
