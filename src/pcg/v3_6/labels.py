"""Evaluator-label firewall.

SignalMatchedFusion legitimately needs harm labels to FIT. The frozen design
requires physical evaluator-label absence during certification. These are two
different record populations and they are kept physically separate, not
"present but ignored".
"""
from typing import Any, Dict, Iterable, List

EVALUATOR_LABEL_KEYS = frozenset({
    "harm_label", "gold_label", "gold", "gold_answer", "evaluator_label",
    "reference", "reference_answer", "reference_judgment", "ground_truth",
    "hidden_gold", "correctness_label", "label_true", "is_harmful",
})
FIT_REQUIRED_LABEL = "harm_label"


class LabelFirewallViolation(RuntimeError):
    pass


def _walk_keys(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path + "." + str(k), str(k)
            yield from _walk_keys(v, path + "." + str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def assert_no_evaluator_labels(record: Any, where: str) -> None:
    """Certification / DEV candidate records must carry no evaluator labels."""
    hits = [p for p, k in _walk_keys(record) if k.lower() in EVALUATOR_LABEL_KEYS]
    if hits:
        raise LabelFirewallViolation(f"EVALUATOR_LABEL_IN_{where}:{sorted(hits)[:8]}")


def assert_fit_labels(records: Iterable[Dict[str, Any]]) -> int:
    """FIT records must each carry a valid binary harm_label. No defaulting."""
    n = 0
    for i, r in enumerate(records):
        if FIT_REQUIRED_LABEL not in r:
            raise LabelFirewallViolation(f"FIT_RECORD_MISSING_HARM_LABEL:index={i}")
        v = r[FIT_REQUIRED_LABEL]
        if isinstance(v, bool):
            v = int(v)
        if v not in (0, 1):
            raise LabelFirewallViolation(f"FIT_RECORD_INVALID_HARM_LABEL:index={i}:value={v!r}")
        n += 1
    if n == 0:
        raise LabelFirewallViolation("FIT_PARTITION_EMPTY")
    return n


def strip_for_certification(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with every evaluator-label key physically removed."""
    if not isinstance(record, dict):
        return record
    out = {}
    for k, v in record.items():
        if str(k).lower() in EVALUATOR_LABEL_KEYS:
            continue
        out[k] = strip_for_certification(v) if isinstance(v, dict) else (
            [strip_for_certification(x) if isinstance(x, dict) else x for x in v]
            if isinstance(v, list) else v)
    return out
