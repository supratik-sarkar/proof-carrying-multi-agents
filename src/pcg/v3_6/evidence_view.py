"""Deterministic prover-native evidence view, byte-identical for all verifiers."""
from typing import Any, Dict, List, Mapping

from .hashing36 import sha256_json, sha256_text
from .labels import assert_no_evaluator_labels

EVIDENCE_VIEW_VERSION = "PCG_MAS_V3_6_EVIDENCE_VIEW_V1"


class EvidenceViewError(RuntimeError):
    pass


def _slot_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for k in ("text", "evidence_text", "content"):
            if isinstance(item.get(k), str):
                return item[k]
    t = getattr(item, "text", None)
    return t if isinstance(t, str) else str(item)


def build_evidence_view(*, model_id: str, dataset_id: str, observation_id: str,
                        retrieved: List[Any], candidate_text: str) -> Dict[str, Any]:
    """One deterministic serialization. Every verifier scores the same bytes."""
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        raise EvidenceViewError(f"EMPTY_CANDIDATE:{observation_id}")
    parts = [_slot_text(x) for x in retrieved]
    if not any(p.strip() for p in parts):
        raise EvidenceViewError(f"EMPTY_EVIDENCE:{observation_id}")
    view_text = "\n\n".join(f"[{i+1}] {p.strip()}" for i, p in enumerate(parts) if p.strip())
    rec = {
        "schema": EVIDENCE_VIEW_VERSION,
        "model_id": model_id, "dataset_id": dataset_id, "observation_id": observation_id,
        "evidence_view_text": view_text,
        "candidate_text": candidate_text,
        "slot_count": sum(1 for p in parts if p.strip()),
        "evidence_view_sha256": sha256_text(view_text),
        "candidate_sha256": sha256_text(candidate_text),
    }
    rec["record_sha256"] = sha256_json({k: v for k, v in rec.items() if k != "record_sha256"})
    assert_no_evaluator_labels(rec, "EVIDENCE_VIEW")
    return rec


def assert_view_identity(views: List[Mapping[str, Any]]) -> bool:
    """All verifiers must receive one view per observation."""
    by = {}
    for v in views:
        k = (v["model_id"], v["dataset_id"], v["observation_id"])
        h = v["evidence_view_sha256"]
        if k in by and by[k] != h:
            raise EvidenceViewError(f"EVIDENCE_VIEW_DIVERGENCE:{k}")
        by[k] = h
    return True
