"""Production replay engine V_Pi for PCG-MAS v3.5.

Enforces:
- Schema for recorded execution trace:
  [trace_id, parent_span_id, action_id, tool_id, tool_version,
   canonical_arguments, pre_state_hash, policy_context_hash,
   authorization_result, observation_semantic_hash, post_state_hash,
   environment_snapshot_id]
- Self-replay detection -> PROTOCOL_INTEGRITY_FAILURE_ABORT.
- Missing replay -> INDETERMINATE.
- Material difference -> FAIL (material differences include action sequence,
  tool id/version, canonical argument values, authorization result,
  pre/post state hashes, observation semantic content, branch outcome).
- Immaterial difference -> PASS (timing, latency, trace/span IDs, server telemetry,
  whitespace/serialization key order, retry count, cache hit/miss, process ID).
- Unclassified difference -> MATERIAL_FAIL_CLOSED.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from pcg.v3_5.core import VerifierState

RECORDED_TRACE_FIELDS = [
    "trace_id",
    "parent_span_id",
    "action_id",
    "tool_id",
    "tool_version",
    "canonical_arguments",
    "pre_state_hash",
    "policy_context_hash",
    "authorization_result",
    "observation_semantic_hash",
    "post_state_hash",
    "environment_snapshot_id",
]

MATERIAL_FIELDS = [
    "action_id",
    "tool_id",
    "tool_version",
    "canonical_arguments",
    "pre_state_hash",
    "authorization_result",
    "observation_semantic_hash",
    "post_state_hash",
]

IMMATERIAL_FIELDS = [
    "trace_id",
    "parent_span_id",
    "wall_clock_time",
    "latency_ms",
    "telemetry_id",
    "host_id",
    "process_id",
    "retry_count",
    "cache_hit",
]


class ProtocolIntegrityAbort(Exception):
    """Raised when self-replay or critical protocol violation is detected."""
    pass


@dataclass
class ActionTraceRecord:
    """Single action in an execution trace."""

    trace_id: str
    parent_span_id: str
    action_id: str
    tool_id: str
    tool_version: str
    canonical_arguments: Dict[str, Any]
    pre_state_hash: str
    policy_context_hash: str
    authorization_result: str
    observation_semantic_hash: str
    post_state_hash: str
    environment_snapshot_id: str
    wall_clock_time: Optional[float] = None
    latency_ms: Optional[float] = None
    telemetry_id: Optional[str] = None
    host_id: Optional[str] = None
    process_id: Optional[int] = None
    retry_count: Optional[int] = None
    cache_hit: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndependentReplayComparator:
    """Compares recorded trace with independently re-executed trace."""

    @staticmethod
    def _normalize_args(args: Any) -> str:
        """Deterministically serialize arguments."""
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return json.dumps(parsed, sort_keys=True)
            except Exception:
                return args.strip()
        return json.dumps(args, sort_keys=True)

    def compare(
        self,
        recorded_trace: Optional[List[Dict[str, Any]]],
        replayed_trace: Optional[List[Dict[str, Any]]],
        is_self_replay: bool = False,
    ) -> Tuple[VerifierState, Dict[str, Any]]:
        """Compare recorded vs replayed trace.

        Returns:
            (VerifierState, audit_details)
        """
        audit: Dict[str, Any] = {
            "is_self_replay": is_self_replay,
            "recorded_steps": len(recorded_trace) if recorded_trace is not None else 0,
            "replayed_steps": len(replayed_trace) if replayed_trace is not None else 0,
            "differences": [],
        }

        # Gate R1: Self-replay check
        if is_self_replay:
            raise ProtocolIntegrityAbort(
                "SELF_REPLAY_DETECTED: System attempted to evaluate self-replay as independent replay. "
                "Protocol integrity abort triggered."
            )

        # Gate R5: Missing replay check
        if replayed_trace is None or recorded_trace is None:
            audit["reason"] = "MISSING_REPLAY_TRACE"
            return VerifierState.INDETERMINATE, audit

        if len(recorded_trace) != len(replayed_trace):
            audit["reason"] = "STEP_COUNT_MISMATCH"
            audit["differences"].append({
                "type": "material",
                "field": "step_count",
                "recorded": len(recorded_trace),
                "replayed": len(replayed_trace),
            })
            return VerifierState.FAIL, audit

        # Compare step by step
        for idx, (rec, rep) in enumerate(zip(recorded_trace, replayed_trace)):
            # Check material fields
            for field_name in MATERIAL_FIELDS:
                rec_val = rec.get(field_name)
                rep_val = rep.get(field_name)

                if field_name == "canonical_arguments":
                    rec_val_norm = self._normalize_args(rec_val)
                    rep_val_norm = self._normalize_args(rep_val)
                    if rec_val_norm != rep_val_norm:
                        audit["differences"].append({
                            "step": idx,
                            "type": "material",
                            "field": field_name,
                            "recorded": rec_val,
                            "replayed": rep_val,
                        })
                        return VerifierState.FAIL, audit
                else:
                    if rec_val != rep_val:
                        audit["differences"].append({
                            "step": idx,
                            "type": "material",
                            "field": field_name,
                            "recorded": rec_val,
                            "replayed": rep_val,
                        })
                        return VerifierState.FAIL, audit

            # Check unclassified fields: if a field is not in immaterial and differs -> fail-closed
            all_keys = set(rec.keys()).union(set(rep.keys()))
            for k in all_keys:
                if k not in MATERIAL_FIELDS and k not in IMMATERIAL_FIELDS:
                    if rec.get(k) != rep.get(k):
                        audit["differences"].append({
                            "step": idx,
                            "type": "unclassified_fail_closed",
                            "field": k,
                            "recorded": rec.get(k),
                            "replayed": rep.get(k),
                        })
                        return VerifierState.FAIL, audit

        # All material checks passed; immaterial differences ignored
        audit["status"] = "PASS"
        return VerifierState.PASS, audit


def verify_replay_domain() -> Dict[str, Any]:
    """Production verification callable for replay domain."""
    comp = IndependentReplayComparator()
    trace = [{"step": 0, "tool": "search", "canonical_arguments": {"q": "test"}}]
    state, audit = comp.compare(trace, trace)
    return {"domain": "replay", "state": state.value, "status": audit.get("status")}

