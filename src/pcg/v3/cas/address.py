"""PCG-CAS-v1 content addressing.

Address format is VERSIONED so a future change cannot silently reinterpret an
existing root. The encoding is unambiguous by construction:

  * a global domain prefix pins the scheme version;
  * every object carries an object-type domain-separation tag, so an object of
    one type can never be reinterpreted as another;
  * every variable-length component is length-prefixed with a fixed-width
    unsigned big-endian length, so concatenation is never ambiguous
    (``[a, b]`` and ``[ab]`` cannot collide);
  * composite children are emitted in a deterministic, semantically defined
    order declared per object type -- never in dict iteration order.

Only COMMITTED fields enter an address. Observed/telemetry fields never do:
a latency or span-id change must not move the certificate root.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..canon import canonical_json

SCHEME = "PCG-CAS-v1"
_GLOBAL_PREFIX = SCHEME.encode("ascii")
_LEN_WIDTH = 8                      # fixed-width unsigned big-endian
_HASH = "sha256"


class ObjType(str, Enum):
    """Domain-separation tags. Never reuse or renumber a tag."""
    CLAIM = "claim"
    EVIDENCE = "evidence"
    EVIDENCE_COLLECTION = "evidence_collection"
    SEMANTIC_TRANSCRIPT = "semantic_transcript"
    PROVIDER_RESULT = "provider_semantic_result"
    RETRIEVAL_RESULT = "retrieval_result"
    TOOL_RESULT = "tool_result"
    POLICY_DECISION = "policy_decision"
    CHECKER_DECISION = "checker_decision"
    REPLAY_INTERVENTION = "replay_intervention"
    CERTIFICATE = "certificate"
    LINEAGE = "lineage_object"
    PER_EXAMPLE_RECORD = "per_example_record"
    EXECUTION_STEP = "execution_step"


#: Deterministic child ordering per object type. Children are addressed in this
#: declared order; a type with children MUST appear here.
CHILD_ORDER: Dict[ObjType, Tuple[str, ...]] = {
    ObjType.EVIDENCE_COLLECTION: ("members",),
    ObjType.SEMANTIC_TRANSCRIPT: ("steps",),
    ObjType.CERTIFICATE: ("claim", "support", "pipeline", "contract",
                          "checker_decision", "policy_decision"),
    ObjType.CHECKER_DECISION: ("inputs",),
    ObjType.POLICY_DECISION: ("inputs",),
    ObjType.REPLAY_INTERVENTION: ("target",),
    ObjType.LINEAGE: ("src", "dst"),
}


def _lp(b: bytes) -> bytes:
    """Length-prefix a component. Fixed width; big-endian; unambiguous."""
    n = len(b)
    if n >= 1 << (8 * _LEN_WIDTH):
        raise ValueError("component too large to length-prefix")
    return n.to_bytes(_LEN_WIDTH, "big") + b


class Address(str):
    """`PCG-CAS-v1:<objtype>:<hex>` -- self-describing and version-pinned."""

    __slots__ = ()

    @property
    def scheme(self) -> str:
        return self.split(":", 2)[0]

    @property
    def objtype(self) -> str:
        return self.split(":", 2)[1]

    @property
    def digest(self) -> str:
        return self.split(":", 2)[2]

    @classmethod
    def parse(cls, s: str) -> "Address":
        parts = s.split(":", 2)
        if len(parts) != 3 or parts[0] != SCHEME:
            raise ValueError(f"not a {SCHEME} address: {s!r}")
        if parts[1] not in {t.value for t in ObjType}:
            raise ValueError(f"unknown object type tag: {parts[1]!r}")
        return cls(s)


def addr(objtype: ObjType, committed: Mapping[str, Any],
         children: Optional[Mapping[str, Sequence["Address"]]] = None) -> Address:
    """Content address of a committed object.

    `committed` must already exclude observed/telemetry fields; `split_fields`
    in `pcg.v3.cas.fields` is the enforcement point.
    """
    if not isinstance(objtype, ObjType):
        raise TypeError("objtype must be an ObjType (domain-separation tag)")
    h = hashlib.new(_HASH)
    h.update(_lp(_GLOBAL_PREFIX))
    h.update(_lp(objtype.value.encode("utf-8")))
    h.update(_lp(canonical_json(dict(committed)).encode("utf-8")))

    order = CHILD_ORDER.get(objtype, ())
    kids = dict(children or {})
    unknown = set(kids) - set(order)
    if unknown:
        raise ValueError(f"{objtype.value}: undeclared child slots {sorted(unknown)}; "
                         "declare them in CHILD_ORDER so ordering is deterministic")
    # slot count, then per-slot: slot name, child count, then each child address
    h.update(_lp(str(len(order)).encode("ascii")))
    for slot in order:
        vals = list(kids.get(slot, []) or [])
        h.update(_lp(slot.encode("utf-8")))
        h.update(_lp(str(len(vals)).encode("ascii")))
        for a in vals:
            h.update(_lp(str(a).encode("ascii")))
    return Address(f"{SCHEME}:{objtype.value}:{h.hexdigest()}")


def child_addresses(objtype: ObjType,
                    children: Mapping[str, Sequence[Address]]) -> List[Address]:
    """Flatten children in the declared deterministic order."""
    out: List[Address] = []
    for slot in CHILD_ORDER.get(objtype, ()):
        out.extend(children.get(slot, []) or [])
    return out
