"""GET /api/v3/runs/{run_id}/execution-graph -- unified reconstruction.

One endpoint returns the whole reviewer-relevant object graph for a run:
certificate closure, the CAS objects reachable from the certificate root, the
checkpoint lineage, the replay links, the artifact lineage edges and the
per-example records -- with the *authoritativeness* of each part declared.

Design rules enforced here:

* Availability is three-state. A run whose certificate root is not closed under
  the store returns ``INDETERMINATE``; it never returns a partial graph as if
  it were complete.
* Records come from the authoritative JSONL store. If a derived view is offered
  it must carry its reconciliation status; the payload states which source was
  actually read.
* Telemetry is returned in a separate ``observed`` block and is explicitly
  labelled non-address-forming, so a consumer cannot mistake it for content
  that participates in the certificate root.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..canon import canonical_json, sha256_text
from ..cas.closure import verify_closed
from ..cas.replay import reconstruct_replay
from ..cas.store import ContentAddressedStore
from ..exec.modes import REPRODUCTION_ADMISSIBLE, ExecutionMode
from ..lineage.edges import LineageGraph

GRAPH_VERSION = "PCG-EXECGRAPH-v1"


class ExecutionGraphError(RuntimeError):
    pass


def _mode_block(mode: Optional[str]) -> Dict[str, Any]:
    if mode is None:
        return {"execution_mode": None, "reproduction_admissible": False,
                "reason": "execution mode not recorded"}
    try:
        m = ExecutionMode(mode)
    except ValueError:
        return {"execution_mode": mode, "reproduction_admissible": False,
                "reason": "unrecognised execution mode"}
    ok = m in REPRODUCTION_ADMISSIBLE
    return {"execution_mode": m.value, "reproduction_admissible": ok,
            "reason": ("admissible as reproduction evidence" if ok else
                       "only FRESH and REPLICATE are admissible as reproduction evidence")}


def execution_graph(run_id: str,
                    certificate_root: Optional[str],
                    cas: ContentAddressedStore,
                    *,
                    spec_hash: str = "",
                    execution_mode: Optional[str] = None,
                    checkpoint_lineage: Sequence[Mapping[str, Any]] = (),
                    lineage: Optional[LineageGraph] = None,
                    records: Sequence[Mapping[str, Any]] = (),
                    record_source: str = "authoritative_jsonl",
                    reconciliation: Optional[Mapping[str, Any]] = None,
                    include_objects: bool = True) -> Dict[str, Any]:
    """Build the execution-graph payload. Pure; performs no I/O beyond `cas`."""
    if not run_id:
        raise ExecutionGraphError("run_id is required")

    if certificate_root is None:
        closure = {"status": "INDETERMINATE", "closed": False,
                   "reason": "no certificate root recorded for this run",
                   "n_objects": 0, "missing": []}
        objects: Dict[str, Any] = {}
        replay: Dict[str, Any] = {"status": "INDETERMINATE", "links": []}
    else:
        rep = verify_closed(str(certificate_root), cas)
        closure = {"status": rep.status, "closed": bool(rep.closed),
                   "n_objects": rep.n_reachable,
                   "missing": sorted(rep.missing), "corrupt": sorted(rep.corrupt)}
        r = reconstruct_replay(run_id, str(certificate_root), spec_hash,
                               [dict(x) for x in checkpoint_lineage], cas)
        objects = r.objects if include_objects else {}
        replay = {"status": r.status,
                  "links": [vars(l) if not hasattr(l, "to_dict") else l.to_dict()
                            for l in r.links]}

    lin = lineage or LineageGraph()
    lineage_block: Dict[str, Any] = lin.to_dict()
    try:
        lineage_block["integrity"] = lin.verify()
        lineage_block["integrity_status"] = "VERIFIED"
    except Exception as exc:
        lineage_block["integrity_status"] = "FAILED"
        lineage_block["integrity_error"] = str(exc)

    committed_records: List[Dict[str, Any]] = []
    observed_records: List[Dict[str, Any]] = []
    from ..cas.fields import split_fields
    for rec in records:
        c, o = split_fields(rec)
        c.pop("_record_hash", None)
        committed_records.append(c)
        o["record_id"] = rec.get("record_id")
        observed_records.append(o)

    payload: Dict[str, Any] = {
        "graph_version": GRAPH_VERSION,
        "run_id": run_id,
        "certificate_root": certificate_root,
        "spec_hash": spec_hash or None,
        "execution": _mode_block(execution_mode),
        "closure": closure,
        "replay": replay,
        "lineage": lineage_block,
        "records": {
            "source": record_source,
            "n": len(committed_records),
            "committed": committed_records,
            "reconciliation": dict(reconciliation) if reconciliation else None,
        },
        "observed": {
            "address_forming": False,
            "note": "telemetry; excluded from every content address by construction",
            "records": observed_records,
        },
        "objects": {"n": len(objects), "by_addr": objects} if include_objects
                   else {"n": 0, "by_addr": {}, "omitted": True},
    }
    payload["payload_hash"] = sha256_text(canonical_json(
        {k: v for k, v in payload.items() if k != "payload_hash"}))
    return payload
