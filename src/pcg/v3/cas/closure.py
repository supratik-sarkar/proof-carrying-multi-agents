"""Closure: is every address reachable from the root resolvable?

An unclosed root is INDETERMINATE. It is never ACCEPT: acceptance cannot be a
function of a graph you cannot fully resolve.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from .store import CASCorruption, CASMissing, ContentAddressedStore


@dataclass
class ClosureReport:
    root: str
    closed: bool
    n_reachable: int
    missing: List[str] = field(default_factory=list)
    corrupt: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "CLOSED" if self.closed else "INDETERMINATE"

    def to_dict(self) -> dict:
        return {**vars(self), "status": self.status}


def verify_closed(root: str, cas: ContentAddressedStore) -> ClosureReport:
    missing: List[str] = []
    corrupt: List[str] = []
    seen = set()
    stack = [str(root)]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            cas.get(cur)
        except CASMissing:
            missing.append(cur)
            continue
        except CASCorruption:
            corrupt.append(cur)
            continue
        stack.extend(str(c) for c in cas.children_of(cur))
    return ClosureReport(str(root), not (missing or corrupt), len(seen), missing, corrupt)
