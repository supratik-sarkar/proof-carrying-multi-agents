"""Execution classes: which backends can *structurally* yield empirical evidence.

A mock backend may produce a perfectly-formed record. It must still never be
DIRECT, because its output is not an observation of a model. Eligibility is a
property of the execution channel, decided here, not of record completeness.
"""
from __future__ import annotations

from enum import Enum


class ExecutionClass(str, Enum):
    LOCAL_MODEL = "LOCAL_MODEL"          # real weights, local forward pass
    REMOTE_PROVIDER = "REMOTE_PROVIDER"  # real remote inference endpoint
    MOCK = "MOCK"                        # deterministic stand-in
    TEST_FIXTURE = "TEST_FIXTURE"        # fixture data for tests
    REPLAY = "REPLAY"                    # reconstructed from stored artifacts
    UNKNOWN = "UNKNOWN"                  # provenance not established


#: Only these two can ever yield DIRECT.
DIRECT_ELIGIBLE: frozenset[ExecutionClass] = frozenset(
    {ExecutionClass.LOCAL_MODEL, ExecutionClass.REMOTE_PROVIDER}
)

#: Registry. Extend when a backend is added; unknown names default to UNKNOWN,
#: which is DIRECT-ineligible, so forgetting to register fails closed.
BACKEND_EXECUTION_CLASS: dict[str, ExecutionClass] = {
    "hf_local": ExecutionClass.LOCAL_MODEL,
    "hf_inference": ExecutionClass.REMOTE_PROVIDER,
    "deepseek": ExecutionClass.REMOTE_PROVIDER,
    "openai": ExecutionClass.REMOTE_PROVIDER,
    "anthropic": ExecutionClass.REMOTE_PROVIDER,
    "mock": ExecutionClass.MOCK,
    "mock-llm": ExecutionClass.MOCK,
    "fixture": ExecutionClass.TEST_FIXTURE,
    "replay": ExecutionClass.REPLAY,
}


def execution_class(backend: str | None) -> ExecutionClass:
    if not backend:
        return ExecutionClass.UNKNOWN
    b = str(backend).lower()
    if b.startswith("mock"):
        return ExecutionClass.MOCK
    return BACKEND_EXECUTION_CLASS.get(b, ExecutionClass.UNKNOWN)


def is_direct_eligible(backend: str | None) -> bool:
    return execution_class(backend) in DIRECT_ELIGIBLE
