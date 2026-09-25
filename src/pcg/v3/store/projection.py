"""Canonical tabular projection shared by the authoritative and derived stores.

Reconciliation is only meaningful if both sides agree on *what* is being
compared. The projection fixes that: a closed column list, a normalisation rule
per value type, a per-row digest, and an order-insensitive table digest.

Order-insensitivity is deliberate. A Parquet writer may reorder row groups and a
DuckDB query may return rows in any order without changing the content; making
the digest depend on physical order would raise false divergence alarms. Content
changes, dropped rows and duplicated rows all still change the digest, because
the digest is taken over the *multiset* of row digests (sorted, with the count
and a domain-separated prefix).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..canon import canonical_json, sha256_text

PROJECTION_VERSION = "PCG-PROJ-v1"

#: Closed column list. Adding a column is a projection-version change, never a
#: silent edit, because the version string is domain-separated into the digest.
PROJECTION_COLUMNS: Tuple[str, ...] = (
    "record_id", "run_id", "experiment_id", "system", "provenance_class",
    "metric_version", "schema_version", "config_hash", "spec_hash", "cell_id",
    "dataset", "split", "example_id", "seed",
    "model_id", "model_revision", "backend_type", "provider_route",
    "decoding_config_hash", "prompt_hash", "backend_fingerprint",
    "checker_fingerprint", "claim_id",
    "v_h", "v_pi", "v_gamma", "v_entail", "check", "certificate_hash",
    "policy_eval_status",
    "int_fail", "replay_fail", "drift_fail", "check_fail", "cov_gap",
    "n_channels_fired",
    "execution_mode", "certificate_root", "instrumentation_version",
)

_FLOAT_QUANTUM = 12  # decimal places retained before hashing


def _norm(v: Any) -> Any:
    """Normalise one cell so JSONL and Parquet round-trips agree.

    - ``None`` stays ``None`` (NOT MEASURED never becomes 0 or "").
    - ``bool`` stays bool and is *never* folded into int, because ``True == 1``
      in Python and a Parquet int column silently reading back as 1 must not
      compare equal to a JSONL boolean ``true``.
    - floats are rounded to a fixed quantum so a float32 Parquet column and a
      float64 JSON literal do not diverge on representation alone; NaN and
      infinities are rejected rather than normalised, since the record schema
      forbids them.
    """
    if v is None or isinstance(v, bool) or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("non-finite value in projection")
        return round(v, _FLOAT_QUANTUM) + 0.0
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _norm(x) for k, x in sorted(v.items())}
    return str(v)


def project_record(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """Restrict a record to the projection columns, normalised."""
    return {c: _norm(rec.get(c)) for c in PROJECTION_COLUMNS}


def row_digest(rec: Mapping[str, Any]) -> str:
    row = project_record(rec)
    return sha256_text(PROJECTION_VERSION + "\x1f" + canonical_json(row))


def projection_digest(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Order-insensitive digest over the multiset of projected rows."""
    digests: List[str] = sorted(row_digest(r) for r in records)
    h = hashlib.sha256()
    h.update(PROJECTION_VERSION.encode())
    h.update(len(digests).to_bytes(8, "big"))
    for d in digests:
        h.update(bytes.fromhex(d))
    return {
        "projection_version": PROJECTION_VERSION,
        "columns": list(PROJECTION_COLUMNS),
        "n_rows": len(digests),
        "digest": h.hexdigest(),
        "row_digests": digests,
    }


def divergent_rows(left: Sequence[Mapping[str, Any]],
                   right: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Localise a divergence to record_ids rather than reporting 'not equal'."""
    lmap = {row_digest(r): r.get("record_id") for r in left}
    rmap = {row_digest(r): r.get("record_id") for r in right}
    only_left = sorted(str(lmap[d]) for d in lmap.keys() - rmap.keys())
    only_right = sorted(str(rmap[d]) for d in rmap.keys() - lmap.keys())
    return {"only_authoritative": only_left, "only_derived": only_right}
