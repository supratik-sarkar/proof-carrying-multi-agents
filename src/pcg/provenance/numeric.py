"""Measured quantities. An unknown measurement is None and stays None.

The rule this module exists to enforce: **no unknown numerical measurement may
become zero.** `(input_tokens or 0) + (output_tokens or 0)` silently converts
"not reported by the provider" into "the provider reported zero", which is a
different and false statement.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

Number = Optional[float]


class UnknownMeasurement(ValueError):
    """Raised when code tries to coerce an unknown measurement into a number."""


def add(*vals: Optional[float]) -> Optional[float]:
    """Sum, or None if ANY operand is unknown. Never treats None as 0."""
    if any(v is None for v in vals):
        return None
    return sum(vals)  # type: ignore[arg-type]


def ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """num/den, or None when either is unknown or the denominator is empty."""
    if num is None or den is None or den == 0:
        return None
    return num / den


def mean(vals: Iterable[Optional[float]], *, require_all: bool = False) -> Optional[float]:
    """Mean of known values. With require_all, any unknown makes the result unknown."""
    vals = list(vals)
    if require_all and any(v is None for v in vals):
        return None
    known = [v for v in vals if v is not None]
    return (sum(known) / len(known)) if known else None


def is_finite_nonneg(v: Optional[float]) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def require(v: Optional[float], field: str) -> float:
    if v is None:
        raise UnknownMeasurement(f"{field} is unknown; it must not be defaulted to a number")
    return v
