"""Committed vs observed field partition -- the enforcement point (D2).

COMMITTED fields determine certificate identity. OBSERVED fields are telemetry
and MUST NOT enter any address: changing latency, host, span ids, cache state,
billed cost or physical retry count must leave the certificate root unchanged.

Note the retry rule: if a retry produced the final semantic response, the
SELECTED RESPONSE CONTENT is committed; the fact that it arrived on physical
attempt 2 is observed. The retry POLICY is committed, but through `spec_hash`.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

#: Telemetry. Never address-forming.
OBSERVED: frozenset = frozenset({
    "timestamp", "started_at", "ended_at", "wall_clock", "captured_utc",
    "latency_ms", "latency_p50_ms", "latency_p95_ms", "phase_timings_ms",
    "duration_ms", "host_fingerprint", "hardware_fingerprint", "device",
    "trace_id", "span_id", "root_span_id", "parent_span_id", "replay_trace_id",
    "langsmith_run_id", "cache_state", "cache_hit", "billed_cost_usd",
    "cost_usd", "retry_attempt", "retry_count", "retry_timings_ms",
    "exporter_latency_ms", "process_id", "pid", "uptime_s",
    "instrumentation_version", "otel_schema_version",
    # bare aliases: a plain `host` or `worker` key must not become
    # address-forming merely because it lacks the `_fingerprint` suffix.
    "host", "hostname", "worker_host", "worker_id", "node_name", "container_id",
    "queue_wait_ms", "tokens_per_second", "throughput_tps", "gpu_name",
    "observed_at", "emitted_at", "exporter_status",
})

#: Semantic. Address-forming where present.
COMMITTED_HINT: frozenset = frozenset({
    "claim", "claim_id", "claim_text", "evidence_ids", "evidence_hashes",
    "evidence_relationships", "semantic_transcript", "selected_output",
    "selected_output_sha256", "raw_output_sha256", "retrieval_digests",
    "tool_result_digests", "prompt_hash", "spec_hash", "seed", "experiment_seed",
    "checker_config_hash", "checker_input_hash", "checker_output",
    "checker_verdict", "checker_fingerprint", "policy_bundle_hash",
    "policy_input_hash", "policy_decision", "policy_input", "tool_schema_hash",
    "v_h", "v_pi", "v_gamma", "v_entail", "controller_action",
    "model_id", "model_revision", "tokenizer_id", "decoding_config_hash",
    "dataset", "split", "example_id", "system", "cell_id",
})


class ObservedFieldInAddress(ValueError):
    """Raised when telemetry would have entered a content address."""


def split_fields(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Partition a payload into (committed, observed)."""
    committed: Dict[str, Any] = {}
    observed: Dict[str, Any] = {}
    for k, v in payload.items():
        (observed if k in OBSERVED else committed)[k] = v
    return committed, observed


def assert_no_observed(committed: Mapping[str, Any]) -> None:
    """Fail loudly rather than silently hashing telemetry."""
    leaked = sorted(set(committed) & OBSERVED)
    if leaked:
        raise ObservedFieldInAddress(
            f"observed/telemetry fields would enter a content address: {leaked}. "
            "Telemetry must not change certificate identity.")


def committed_of(payload: Mapping[str, Any]) -> Dict[str, Any]:
    c, _ = split_fields(payload)
    assert_no_observed(c)
    return c
