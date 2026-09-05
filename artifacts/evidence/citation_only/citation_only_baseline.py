#!/usr/bin/env python3
"""Scope-matched Citation-Only baseline.

Acceptance predicate: accept iff citation validity AND entailment both pass.
No replay, no execution contract, no redundancy, no controller — that exclusion
IS the baseline. Pure predicate; contains no metric outcome.

PROVENANCE: PROTOCOL. This module defines a decision rule. Any number produced by
applying it to the 56-cell records inherits that record set's provenance
(UNKNOWN_WITH_GENERATIVE_INDICATORS) and is not empirical.
"""
from __future__ import annotations

PROVENANCE_CLASS = "PROTOCOL"
REQUIRED_FIELDS = ("has_citation", "entails")
EXCLUDED_FEATURES = ("replay", "execution_contract", "redundancy", "controller")


class MissingEvidenceError(KeyError):
    """Raised when a required field is absent. Never defaulted."""


def evaluate_citation_only(record: dict) -> bool:
    """Return the Citation-Only acceptance decision for one record.

    Fail-closed: a missing field raises. It is never treated as False, because
    'absent' and 'observed false' are different states.
    """
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}")
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise MissingEvidenceError(
            f"Citation-Only requires {REQUIRED_FIELDS}; missing {missing}. "
            "Fail-closed: no default substituted."
        )
    leaked = [f for f in EXCLUDED_FEATURES if record.get(f)]
    if leaked:
        raise ValueError(
            f"Scope violation: Citation-Only must not consume {leaked}. "
            "The baseline is defined by excluding these features."
        )
    return bool(record["has_citation"]) and bool(record["entails"])


def evaluate_many(records) -> list[bool]:
    return [evaluate_citation_only(r) for r in records]


def acceptance_rate(records):
    """Accepted fraction, or None when there is nothing to divide by."""
    d = list(records)
    return (sum(evaluate_many(d)) / len(d)) if d else None
