"""Artifact lineage as content-addressed edges.

An edge is itself a committed object: its address is derived from the endpoints'
addresses, the relation tag and the spec hash. That has two consequences a flat
provenance log cannot offer.

1. An edge cannot be silently retargeted. Changing ``src_addr``, ``dst_addr``,
   ``relation`` or ``spec_hash`` changes the edge address, so a lineage graph
   pinned by its root hash pins the *shape* of the derivation, not just its
   node contents.
2. Because endpoints are addresses rather than mutable identifiers, an edge
   referring to an artifact that was later modified simply fails to resolve --
   it cannot accidentally point at the new version.

The graph exposes `ancestors` / `descendants` for the reviewer question "which
artifact supported which claim", and `verify` for "is this graph internally
consistent and closed under the store I have".
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..canon import canonical_json, sha256_text
from ..cas.address import Address, ObjType, addr


class Relation(str, Enum):
    """Closed relation vocabulary. Adding a member is a schema change."""
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CONSUMED_BY = "consumed_by"
    VALIDATED_BY = "validated_by"
    DELEGATED_TO = "delegated_to"
    REPLAY_OF = "replay_of"
    SUMMARISES = "summarises"


class LineageIntegrityError(RuntimeError):
    """A lineage graph failed a structural or content-address check."""


@dataclass(frozen=True)
class ArtifactLineageEdge:
    src_addr: str
    dst_addr: str
    relation: Relation
    spec_hash: str

    @property
    def committed(self) -> Dict[str, Any]:
        return {"relation": self.relation.value, "spec_hash": self.spec_hash}

    @property
    def address(self) -> Address:
        return addr(ObjType.LINEAGE, self.committed,
                    {"src": [self.src_addr], "dst": [self.dst_addr]})

    def to_dict(self) -> Dict[str, Any]:
        return {"edge_addr": str(self.address), "src_addr": self.src_addr,
                "dst_addr": self.dst_addr, "relation": self.relation.value,
                "spec_hash": self.spec_hash}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ArtifactLineageEdge":
        e = ArtifactLineageEdge(str(d["src_addr"]), str(d["dst_addr"]),
                                Relation(d["relation"]), str(d["spec_hash"]))
        declared = d.get("edge_addr")
        if declared is not None and declared != str(e.address):
            raise LineageIntegrityError(
                f"edge address does not bind its endpoints: declared {declared}")
        return e


class LineageGraph:
    """A set of edges plus a root hash over their addresses."""

    def __init__(self, edges: Iterable[ArtifactLineageEdge] = ()):
        self._edges: List[ArtifactLineageEdge] = []
        self._seen: Set[str] = set()
        for e in edges:
            self.add(e)

    def add(self, edge: ArtifactLineageEdge) -> Address:
        a = edge.address
        if str(a) not in self._seen:
            self._seen.add(str(a))
            self._edges.append(edge)
        return a

    @property
    def edges(self) -> List[ArtifactLineageEdge]:
        return list(self._edges)

    @property
    def root(self) -> str:
        """Order-insensitive Merkle root over edge addresses."""
        addrs = sorted(str(e.address) for e in self._edges)
        return sha256_text(canonical_json({"scheme": "PCG-LINEAGE-v1",
                                           "n": len(addrs), "edges": addrs}))

    def nodes(self) -> List[str]:
        s: Set[str] = set()
        for e in self._edges:
            s.add(e.src_addr)
            s.add(e.dst_addr)
        return sorted(s)

    def out_edges(self, node: str) -> List[ArtifactLineageEdge]:
        return [e for e in self._edges if e.src_addr == node]

    def in_edges(self, node: str) -> List[ArtifactLineageEdge]:
        return [e for e in self._edges if e.dst_addr == node]

    def ancestors(self, node: str) -> List[str]:
        return self._walk(node, forward=False)

    def descendants(self, node: str) -> List[str]:
        return self._walk(node, forward=True)

    def _walk(self, node: str, forward: bool) -> List[str]:
        seen: Set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            nxt = ([e.dst_addr for e in self.out_edges(n)] if forward
                   else [e.src_addr for e in self.in_edges(n)])
            for m in nxt:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return sorted(seen)

    def has_cycle(self) -> bool:
        colour: Dict[str, int] = {}

        def visit(n: str) -> bool:
            colour[n] = 1
            for e in self.out_edges(n):
                c = colour.get(e.dst_addr, 0)
                if c == 1 or (c == 0 and visit(e.dst_addr)):
                    return True
            colour[n] = 2
            return False

        return any(colour.get(n, 0) == 0 and visit(n) for n in self.nodes())

    def verify(self, store: Optional[Any] = None,
               allow_cycles: bool = False) -> Dict[str, Any]:
        """Recompute every edge address; optionally require store closure."""
        for e in self._edges:
            recomputed = addr(ObjType.LINEAGE, e.committed,
                              {"src": [e.src_addr], "dst": [e.dst_addr]})
            if str(recomputed) != str(e.address):
                raise LineageIntegrityError("edge address is not reproducible")
        cyclic = self.has_cycle()
        if cyclic and not allow_cycles:
            raise LineageIntegrityError("lineage graph contains a cycle")
        dangling: List[str] = []
        if store is not None:
            dangling = sorted(n for n in self.nodes() if not store.has(n))
            if dangling:
                raise LineageIntegrityError(
                    f"lineage references {len(dangling)} object(s) absent from the store")
        return {"n_edges": len(self._edges), "n_nodes": len(self.nodes()),
                "root": self.root, "cyclic": cyclic, "dangling": dangling}

    def to_dict(self) -> Dict[str, Any]:
        return {"lineage_version": "PCG-LINEAGE-v1", "root": self.root,
                "edges": [e.to_dict() for e in self._edges]}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LineageGraph":
        g = LineageGraph(ArtifactLineageEdge.from_dict(e) for e in d.get("edges", []))
        declared = d.get("root")
        if declared is not None and declared != g.root:
            raise LineageIntegrityError("lineage root does not match its edges")
        return g
