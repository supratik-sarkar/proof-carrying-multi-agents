"""Replay reconstruction from artifacts alone.

Acceptance criterion: given the frozen spec, checkpoint lineage, certificate and
canonical artifacts, the replay path is reconstructible WITHOUT consulting an
opaque mutable Python object from the original run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .address import Address, ObjType
from .closure import verify_closed
from .store import ContentAddressedStore


@dataclass
class ReplayLink:
    source_checkpoint_id: str
    source_span_id: str
    intervention_spec_hash: str
    result_checkpoint_id: str
    replay_trace_id: str
    intervention_addr: Optional[str] = None

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ReconstructedReplay:
    run_id: str
    certificate_root: str
    closed: bool
    spec_hash: str
    links: List[ReplayLink] = field(default_factory=list)
    objects: Dict[str, dict] = field(default_factory=dict)
    reconstructed_from: str = "spec + checkpoint lineage + certificate + CAS"
    status: str = "OK"

    def to_dict(self) -> dict:
        return {**vars(self), "links": [l.to_dict() for l in self.links]}


def reconstruct_replay(run_id: str, certificate_root: str, spec_hash: str,
                       checkpoint_lineage: List[Dict[str, Any]],
                       cas: ContentAddressedStore) -> ReconstructedReplay:
    """Rebuild the replay path purely from persisted artifacts."""
    rep = verify_closed(certificate_root, cas)
    links: List[ReplayLink] = []
    for row in checkpoint_lineage:
        if row.get("kind") != "replay":
            continue
        links.append(ReplayLink(
            source_checkpoint_id=row["source_checkpoint_id"],
            source_span_id=row.get("source_span_id", ""),
            intervention_spec_hash=row["intervention_spec_hash"],
            result_checkpoint_id=row["result_checkpoint_id"],
            replay_trace_id=row.get("replay_trace_id", ""),
            intervention_addr=row.get("intervention_addr"),
        ))
    objects: Dict[str, dict] = {}
    if rep.closed:
        for a in cas.reachable(certificate_root):
            try:
                objects[a] = cas.get(a)
            except Exception:
                pass
    return ReconstructedReplay(
        run_id=run_id, certificate_root=str(certificate_root), closed=rep.closed,
        spec_hash=spec_hash, links=links, objects=objects,
        status="OK" if rep.closed else "INDETERMINATE")
