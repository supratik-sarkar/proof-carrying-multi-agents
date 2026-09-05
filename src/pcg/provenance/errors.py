"""Execution status vocabulary and credential-safe error sanitisation."""
from __future__ import annotations

import re
from enum import Enum


class ExecutionStatus(str, Enum):
    OK = "ok"
    REFUSED = "refused"                  # model declined; a real observation
    PROVIDER_ERROR = "provider_error"    # 5xx / transport / rate limit
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"  # our own schema/contract failure
    INTERRUPTED = "interrupted"          # operator or signal


#: Statuses eligible to contribute to *outcome* metrics (harm, utility, coverage).
#: A refusal IS an observation of model behaviour and is eligible.
OUTCOME_ELIGIBLE: frozenset[ExecutionStatus] = frozenset(
    {ExecutionStatus.OK, ExecutionStatus.REFUSED}
)

#: Statuses that still represent a genuine provider interaction. These are
#: DIRECT *execution evidence* (the call happened) but are INELIGIBLE for
#: outcome metrics, because no outcome was observed. They are counted in
#: attempt/reliability denominators only.
ATTEMPT_ONLY: frozenset[ExecutionStatus] = frozenset(
    {ExecutionStatus.PROVIDER_ERROR, ExecutionStatus.TIMEOUT}
)

_SECRET_PATTERNS = [
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(?:sk|hf|xai|gsk)[-_][A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|apikey|token|secret|password|authorization)\b[\"']?\s*[:=]\s*[\"']?[^\s\"',}]+"),
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),          # long opaque strings
]


def sanitize(msg: str | None, limit: int = 500) -> str | None:
    """Strip anything that could be a credential before an error is persisted."""
    if not msg:
        return None
    out = str(msg)
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out[:limit]
