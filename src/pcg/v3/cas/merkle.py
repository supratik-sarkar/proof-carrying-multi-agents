"""Merkle descent, diff and disagreement localization.

Prop. A2 (disagreement localization) in executable form: if two auditors
disagree on acceptance then either the committed roots differ -- and descent
names the differing leaves -- or the roots agree and the divergence lies in a
TCB component, identified by per-component substitution (A13). Residual
disagreement at fixed root AND fixed TCB is reported EXPLICITLY as
checker-attributable rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .address import Address
from .store import CASCorruption, CASMissing, ContentAddressedStore


@dataclass
class DiffNode:
    path: Tuple[str, ...]
    addr_a: Optional[str]
    addr_b: Optional[str]
    objtype: Optional[str]
    is_leaf: bool

    def to_dict(self) -> dict:
        return {**vars(self), "path": list(self.path)}


def merkle_diff(root_a: str, root_b: str,
                cas_a: ContentAddressedStore,
                cas_b: Optional[ContentAddressedStore] = None,
                _path: Tuple[str, ...] = ("root",)) -> List[DiffNode]:
    """Descend two roots, returning the differing LEAVES.

    Cost is O(d*k) for k differing leaves at depth <= d: identical subtrees are
    pruned on address equality without being read.
    """
    cb = cas_b or cas_a
    if str(root_a) == str(root_b):
        return []                                   # identical subtree: prune
    try:
        ea, eb = cas_a.get(root_a), cb.get(root_b)
    except (CASMissing, CASCorruption):
        return [DiffNode(_path, str(root_a), str(root_b), None, True)]

    ta, tb = ea["objtype"], eb["objtype"]
    ka = {k: [str(x) for x in v] for k, v in ea["children"].items()}
    kb = {k: [str(x) for x in v] for k, v in eb["children"].items()}

    # different type, or different child shape -> this node is the difference
    if ta != tb or sorted(ka) != sorted(kb) or \
       any(len(ka[s]) != len(kb.get(s, [])) for s in ka):
        return [DiffNode(_path, str(root_a), str(root_b), f"{ta}|{tb}", True)]

    if not ka:                                      # leaf with differing payload
        return [DiffNode(_path, str(root_a), str(root_b), ta, True)]

    out: List[DiffNode] = []
    for slot in sorted(ka):
        for i, (ca, cbb) in enumerate(zip(ka[slot], kb[slot])):
            out.extend(merkle_diff(ca, cbb, cas_a, cb, _path + (f"{slot}[{i}]",)))
    if not out:
        # children all equal but payload differs
        out.append(DiffNode(_path, str(root_a), str(root_b), ta, True))
    return out


@dataclass
class Disagreement:
    verdict_a: Optional[bool]
    verdict_b: Optional[bool]
    root_a: str
    root_b: str
    tcb_a: Dict[str, str]
    tcb_b: Dict[str, str]
    cause: str                       # COMMITTED_OBJECT | TCB_COMPONENT | CHECKER_RESIDUAL | NONE
    differing_leaves: List[dict] = field(default_factory=list)
    differing_tcb_components: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return vars(self)


def localize_disagreement(verdict_a: Optional[bool], verdict_b: Optional[bool],
                          root_a: str, root_b: str,
                          tcb_a: Dict[str, str], tcb_b: Dict[str, str],
                          cas_a: ContentAddressedStore,
                          cas_b: Optional[ContentAddressedStore] = None) -> Disagreement:
    """Attribute a disagreement to a named object or a named TCB component."""
    if verdict_a == verdict_b:
        return Disagreement(verdict_a, verdict_b, str(root_a), str(root_b),
                            tcb_a, tcb_b, "NONE", note="verdicts agree")

    if str(root_a) != str(root_b):
        leaves = [d.to_dict() for d in merkle_diff(root_a, root_b, cas_a, cas_b)]
        return Disagreement(verdict_a, verdict_b, str(root_a), str(root_b),
                            tcb_a, tcb_b, "COMMITTED_OBJECT", differing_leaves=leaves,
                            note="committed execution objects differ; see differing_leaves")

    diff_tcb = sorted(k for k in set(tcb_a) | set(tcb_b) if tcb_a.get(k) != tcb_b.get(k))
    if diff_tcb:
        return Disagreement(verdict_a, verdict_b, str(root_a), str(root_b),
                            tcb_a, tcb_b, "TCB_COMPONENT",
                            differing_tcb_components=diff_tcb,
                            note="roots identical; pinned TCB components differ")

    # Same committed root, same pinned TCB, different verdict.
    return Disagreement(
        verdict_a, verdict_b, str(root_a), str(root_b), tcb_a, tcb_b,
        "CHECKER_RESIDUAL",
        note=("same closed root and same pinned TCB specification yet different verdict: "
              "residual is checker non-determinism across host stacks. Reported explicitly; "
              "A04 measures this quantity and the architecture does not presume its value."))
