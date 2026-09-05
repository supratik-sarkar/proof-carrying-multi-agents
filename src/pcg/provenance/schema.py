"""Frozen DIRECT execution-record schema (RC2).

Two rules govern this module:

1. **provenance_class is derived, never supplied.** The recorder rejects any
   caller that sets it.
2. **Unknown is not zero.** Every optional measurement defaults to None and stays
   None; nothing here coerces an absent value into a number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

SCHEMA_ID = "pcg.direct_execution_record/2"

ProvenanceClass = Literal[
    "DIRECT",                        # full native evidence, full usage
    "DIRECT_WITH_PARTIAL_USAGE",     # full native evidence, incomplete token usage
    "DIRECT_ATTEMPT_NO_OUTCOME",     # genuine call that errored/timed out
    "MOCK", "TEST_FIXTURE", "REPLAY",
    "INCOMPLETE",                    # eligible channel, missing evidence
    "UNKNOWN",
]

UsageCompleteness = Literal["FULL", "PARTIAL", "ABSENT"]


@dataclass(frozen=True)
class DecodingParams:
    temperature: float
    top_p: float
    max_tokens: int
    seed: Optional[int] = None
    stop: tuple[str, ...] = ()

    def digest(self) -> str:
        from .hashing import hash_obj
        return hash_obj(asdict(self))


@dataclass(frozen=True)
class ToolCall:
    name: str
    args_hash: str
    output_hash: str
    latency_ms: Optional[float] = None
    in_contract: Optional[bool] = None


@dataclass(frozen=True)
class CheckerOutcomes:
    v_h: Optional[bool] = None
    v_pi: Optional[bool] = None
    v_gamma: Optional[bool] = None
    v_entail: Optional[bool] = None
    entail_score: Optional[float] = None
    accepted: Optional[bool] = None


@dataclass(frozen=True)
class ExecutionEvidence:
    """What a backend must surface. Every field Optional: absent means absent.

    `usage_source` records HOW token counts were obtained — a provider `usage`
    block, a local tokenizer count, or an estimate. Estimates never qualify as
    FULL usage.
    """
    # model identity
    requested_model: Optional[str] = None
    returned_model: Optional[str] = None
    model_revision: Optional[str] = None
    provider: Optional[str] = None
    backend: Optional[str] = None
    endpoint_route: Optional[str] = None      # host/path only; never credentials
    provider_request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    applied_seed: Optional[int] = None

    # timing
    start_timestamp: Optional[str] = None     # ISO-8601 UTC
    end_timestamp: Optional[str] = None
    latency_ms: Optional[float] = None        # from time.monotonic()
    monotonic_measured: Optional[bool] = None

    # usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    usage_source: Optional[Literal["provider_usage", "local_tokenizer", "estimate"]] = None

    # runtime identity
    tokenizer_id: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    library_versions: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DirectExecutionRecord:
    schema_id: str
    record_id: str
    run_id: str
    identity: dict                     # RecordIdentity.canonical()
    identity_digest: str

    evidence: dict                     # ExecutionEvidence.to_dict()
    decoding_params: Optional[dict]

    input_hash: Optional[str]
    output_hash: Optional[str]
    bundle_ref: Optional[str]          # relative path to the execution bundle
    bundle_hash: Optional[str]

    model_call_count: Optional[int] = None
    tool_call_trace: tuple[ToolCall, ...] = ()
    checker_outcomes: CheckerOutcomes = field(default_factory=CheckerOutcomes)
    certificate_hash: Optional[str] = None
    support_hashes: tuple[str, ...] = ()

    status: str = "ok"
    error_type: Optional[str] = None
    error_message: Optional[str] = None     # sanitised
    refusal_reason: Optional[str] = None

    # derived — see classify.py
    provenance_class: Optional[str] = None
    provenance_reasons: tuple[str, ...] = ()
    execution_class: Optional[str] = None
    usage_completeness: Optional[str] = None
    outcome_eligible: Optional[bool] = None

    code_fingerprint: Optional[str] = None
    git_commit: Optional[str] = None
    env_fingerprint: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def json_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "type": "object",
        "required": ["schema_id", "record_id", "run_id", "identity", "identity_digest",
                     "evidence", "status", "provenance_class", "execution_class"],
        "properties": {
            "schema_id": {"const": SCHEMA_ID},
            "provenance_class": {"enum": list(ProvenanceClass.__args__)},
            "usage_completeness": {"enum": ["FULL", "PARTIAL", "ABSENT", None]},
            "latency_ms": {"type": ["number", "null"], "minimum": 0},
        },
    }
