"""Canonical aggregation: the single numerical source of truth.

Replaces the removed "no constant columns" gate. That gate was wrong — a
legitimately measured quantity may be constant. What actually needs enforcing is
*lineage*: every reported number must be recomputable from eligible per-example
records, and no numeric literal may enter the reporting path.

An AggregateResult therefore carries the ids and hash of the exact record set it
was computed from, so `verify_recomputable()` can re-derive it later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional

from .hashing import hash_obj
from .numeric import ratio

AGG_VERSION = "pcg-aggregate/1"

#: Classes eligible to contribute to OUTCOME metrics (harm, utility, coverage).
OUTCOME_ELIGIBLE_CLASSES = frozenset({"DIRECT", "DIRECT_WITH_PARTIAL_USAGE"})
#: Classes eligible to contribute to COST metrics (tokens, throughput).
COST_ELIGIBLE_CLASSES = frozenset({"DIRECT"})
#: Never eligible for any empirical claim.
FORBIDDEN_CLASSES = frozenset(
    {"MOCK", "TEST_FIXTURE", "REPLAY", "UNKNOWN", "INCOMPLETE",
     "DIRECT_ATTEMPT_NO_OUTCOME", "DERIVED_FROM_UNKNOWN_PROVENANCE_56_CELL"}
)


class IneligibleRecords(ValueError):
    pass


@dataclass(frozen=True)
class AggregateResult:
    metric: str
    value: Optional[float]
    numerator: Optional[float]
    denominator: Optional[int]
    n_eligible: int
    n_excluded: int
    provenance_class: str = "DERIVED_FROM_DIRECT"
    metric_kind: str = "outcome"           # outcome | cost
    source_record_ids: tuple[str, ...] = ()
    source_record_set_hash: str = ""
    exclusion_reasons: dict = field(default_factory=dict)
    aggregate_version: str = AGG_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def eligible(records: Iterable[dict], kind: str = "outcome") -> tuple[list[dict], dict]:
    allowed = OUTCOME_ELIGIBLE_CLASSES if kind == "outcome" else COST_ELIGIBLE_CLASSES
    keep, why = [], {}
    for r in records:
        cls = r.get("provenance_class")
        if cls not in allowed:
            why[cls] = why.get(cls, 0) + 1
            continue
        if kind == "outcome" and not r.get("outcome_eligible"):
            why["not_outcome_eligible"] = why.get("not_outcome_eligible", 0) + 1
            continue
        keep.append(r)
    return keep, why


def aggregate(records: Iterable[dict], metric: str,
              numerator_fn: Callable[[dict], Optional[float]],
              denominator_fn: Callable[[dict], bool] = lambda r: True,
              kind: str = "outcome") -> AggregateResult:
    """Compute one metric with full lineage. Empty denominator -> value None."""
    records = list(records)
    keep, why = eligible(records, kind)
    den_recs = [r for r in keep if denominator_fn(r)]
    nums = [numerator_fn(r) for r in den_recs]
    if any(n is None for n in nums):
        num_total = None
    else:
        num_total = sum(nums)  # type: ignore[arg-type]
    den = len(den_recs)
    ids = tuple(sorted(r["record_id"] for r in den_recs))
    prov_cls = "DERIVED_FROM_DIRECT" if (keep and all(r.get("provenance_class") in ("DIRECT", "DIRECT_WITH_PARTIAL_USAGE") for r in keep)) else "DERIVED_FROM_BLOCKED_PROVENANCE"
    return AggregateResult(
        metric=metric, value=ratio(num_total, den), numerator=num_total,
        denominator=den if den else None, n_eligible=len(keep),
        n_excluded=len(records) - len(keep), provenance_class=prov_cls, metric_kind=kind,
        source_record_ids=ids, source_record_set_hash=hash_obj(list(ids)),
        exclusion_reasons=why,
    )


def verify_recomputable(agg: AggregateResult, records: Iterable[dict], metric: str,
                        numerator_fn, denominator_fn=lambda r: True,
                        kind: str = "outcome") -> bool:
    """Re-derive the aggregate from the same record set and compare."""
    by_id = {r["record_id"]: r for r in records}
    subset = [by_id[i] for i in agg.source_record_ids if i in by_id]
    if len(subset) != len(agg.source_record_ids):
        return False
    again = aggregate(subset, metric, numerator_fn, denominator_fn, kind)
    return (again.value == agg.value
            and again.source_record_set_hash == agg.source_record_set_hash)
