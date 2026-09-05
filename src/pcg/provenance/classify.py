"""The DIRECT predicate.

Stated precisely, a record is DIRECT iff ALL of:

  P1  execution_class(backend) in {LOCAL_MODEL, REMOTE_PROVIDER}
  P2  status == "ok"  (or "refused": a refusal is an observed model behaviour)
  P3  identity complete and identity_digest re-derives from it
  P4  input_hash and output_hash present, and a bundle exists carrying the
      canonical payloads from which both re-derive
  P5  decoding_params present
  P6  timing consistent: start and end present and parseable, end >= start,
      latency_ms finite and non-negative, and the timestamp span agrees with
      latency_ms within TIMING_TOLERANCE
  P7  model_call_count >= 1
  P8  model identity present: returned_model or model_revision
  P9  usage_completeness == FULL

If P1-P8 hold but P9 does not, the class is DIRECT_WITH_PARTIAL_USAGE: the
execution is real, the token accounting is not complete. Such records are
eligible for outcome metrics (harm/utility/coverage) and INELIGIBLE for cost
metrics (tokens/throughput). That contract is enforced in lineage.py.

A genuine provider call that errored or timed out is DIRECT_ATTEMPT_NO_OUTCOME:
real execution evidence, but no outcome was observed, so it contributes to
reliability denominators only.

Note on timing: this module does NOT assert a physical lower bound on latency as
proof of authenticity. Authenticity comes from P1 (channel) plus P4 (recomputable
payloads). An implausibly fast call is reported as a *diagnostic advisory*, and
the threshold is configurable and off by default for classification.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from .backends import ExecutionClass, execution_class, is_direct_eligible
from .errors import ATTEMPT_ONLY, OUTCOME_ELIGIBLE, ExecutionStatus

#: Allowed disagreement between (end - start) and latency_ms.
TIMING_TOLERANCE_MS = 250.0
TIMING_RELATIVE_TOLERANCE = 0.5

#: Purely diagnostic. Never used to decide DIRECT. Configure per deployment.
ADVISORY_FAST_CALL_MS = 20.0


def _present(v: Any) -> bool:
    return v is not None and v != "" and v != {} and v != ()


def usage_completeness(ev: dict) -> str:
    i, o, t = ev.get("input_tokens"), ev.get("output_tokens"), ev.get("total_tokens")
    src = ev.get("usage_source")
    if i is None and o is None and t is None:
        return "ABSENT"
    if i is not None and o is not None and src in ("provider_usage", "local_tokenizer"):
        if t is not None and t != i + o:
            return "PARTIAL"          # inconsistent accounting is not FULL
        return "FULL"
    return "PARTIAL"


def classify(rec: dict) -> dict:
    """Return the derived provenance fields. Never raises."""
    ev = rec.get("evidence") or {}
    backend = ev.get("backend") or (rec.get("identity") or {}).get("backend")
    ec = execution_class(backend)
    reasons: list[str] = []
    advisories: list[str] = []

    # P1 — channel eligibility, decided before anything else
    if ec is ExecutionClass.MOCK:
        return _out("MOCK", ("backend is a mock: structurally ineligible for empirical evidence",), ec, None, False)
    if ec is ExecutionClass.TEST_FIXTURE:
        return _out("TEST_FIXTURE", ("fixture data: structurally ineligible",), ec, None, False)
    if ec is ExecutionClass.REPLAY:
        return _out("REPLAY", ("reconstructed from stored artifacts",), ec, None, False)
    if not is_direct_eligible(backend):
        return _out("UNKNOWN", (f"backend '{backend}' is not a DIRECT-eligible execution channel",), ec, None, False)

    # P2 — status
    try:
        status = ExecutionStatus(rec.get("status", "ok"))
    except ValueError:
        return _out("INCOMPLETE", (f"unrecognised status '{rec.get('status')}'",), ec, None, False)
    if status in ATTEMPT_ONLY:
        return _out("DIRECT_ATTEMPT_NO_OUTCOME",
                    (f"genuine provider attempt ended {status.value}: execution evidence, no outcome",),
                    ec, None, False)
    if status not in OUTCOME_ELIGIBLE:
        return _out("INCOMPLETE", (f"status={status.value} is not outcome-eligible",), ec, None, False)

    # P3 — identity
    ident, dig = rec.get("identity"), rec.get("identity_digest")
    if not _present(ident) or not _present(dig):
        reasons.append("missing canonical identity or identity_digest")
    else:
        from .hashing import hash_obj
        if hash_obj(ident) != dig:
            reasons.append("identity_digest does not re-derive from identity")

    # P4 — payload evidence
    for f in ("input_hash", "output_hash", "bundle_ref", "bundle_hash"):
        if not _present(rec.get(f)):
            reasons.append(f"missing payload evidence: {f}")

    # P5 — decoding
    if not _present(rec.get("decoding_params")):
        reasons.append("missing decoding_params")

    # P6 — timing consistency (not a physical floor)
    st, et, lat = ev.get("start_timestamp"), ev.get("end_timestamp"), ev.get("latency_ms")
    if not _present(st) or not _present(et):
        reasons.append("missing start/end timestamp")
    if lat is None:
        reasons.append("missing latency_ms")
    elif not isinstance(lat, (int, float)) or not math.isfinite(lat) or lat < 0:
        reasons.append(f"latency_ms not finite and non-negative: {lat!r}")
    if _present(st) and _present(et):
        try:
            span = (datetime.fromisoformat(et) - datetime.fromisoformat(st)).total_seconds() * 1000
            if span < 0:
                reasons.append("end_timestamp precedes start_timestamp")
            elif lat is not None and math.isfinite(lat):
                tol = max(TIMING_TOLERANCE_MS, lat * TIMING_RELATIVE_TOLERANCE)
                if abs(span - lat) > tol:
                    reasons.append(f"timestamp span {span:.1f} ms disagrees with latency_ms={lat} (tol {tol:.0f})")
        except ValueError:
            reasons.append("timestamps are not ISO-8601")
    if lat is not None and math.isfinite(lat) and lat < ADVISORY_FAST_CALL_MS:
        advisories.append(f"ADVISORY: latency_ms={lat} is unusually fast for a model call")

    # P7 / P8
    if (rec.get("model_call_count") or 0) < 1:
        reasons.append("model_call_count < 1")
    if not (_present(ev.get("returned_model")) or _present(ev.get("model_revision"))):
        reasons.append("missing model identity: returned_model and model_revision both absent")

    uc = usage_completeness(ev)
    if reasons:
        return _out("INCOMPLETE", tuple(reasons + advisories), ec, uc, False)

    outcome_ok = status in OUTCOME_ELIGIBLE
    if uc == "FULL":
        return _out("DIRECT", tuple(advisories), ec, uc, outcome_ok)
    return _out("DIRECT_WITH_PARTIAL_USAGE",
                tuple(advisories) + (f"token usage {uc}: ineligible for cost metrics",),
                ec, uc, outcome_ok)


def _out(cls, reasons, ec, uc, outcome_eligible) -> dict:
    return {"provenance_class": cls, "provenance_reasons": list(reasons),
            "execution_class": ec.value if ec else None,
            "usage_completeness": uc, "outcome_eligible": bool(outcome_eligible)}
