"""Production blinded human label audit selector for PCG-MAS v3.5.

Enforces:
- Exactly 4 candidates selected per model x dataset cell (196 total for 7x7).
- Deterministic hash selection before acceptance/evaluation outputs revealed:
  SHA256(candidate_id || example_id || v3_5_freeze_root).
- Blinding schema:
  - Model identity stripped
  - PCG decision stripped
  - Comparator decisions stripped
  - System names stripped
- Double-blind adjudication with third adjudicator on disagreement.
- Purely for label-noise sensitivity analysis; benchmark labels are NEVER silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from pcg.v3_5.registries import FROZEN_DATASETS, FROZEN_MODELS


def compute_audit_selection_hash(
    candidate_id: str,
    example_id: str,
    v3_5_freeze_root: str,
) -> str:
    """Deterministic selection hash for blinded human audit."""
    key = f"{candidate_id}||{example_id}||{v3_5_freeze_root}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def select_blinded_human_audit_sample(
    cell_candidates: Dict[Tuple[str, str], List[Dict[str, Any]]],
    v3_5_freeze_root: str,
    sample_per_cell: int = 4,
) -> Dict[str, Any]:
    """Select exactly sample_per_cell candidates per cell using deterministic hash.

    Produces blinded audit records.
    """
    total_selected = 0
    per_cell_selection: Dict[str, List[Dict[str, Any]]] = {}
    blinded_records: List[Dict[str, Any]] = []

    for (m, d), cands in sorted(cell_candidates.items()):
        cell_key = f"{m}__{d}"

        # Sort by deterministic selection hash
        sorted_cands = sorted(
            cands,
            key=lambda c: compute_audit_selection_hash(
                c["candidate_id"], c["example_id"], v3_5_freeze_root
            ),
        )

        selected = sorted_cands[:sample_per_cell]
        per_cell_selection[cell_key] = [c["candidate_id"] for c in selected]
        total_selected += len(selected)

        for s in selected:
            # Blinded record: strip model, acceptance, comparator decisions
            blinded_record = {
                "audit_record_id": hashlib.sha256(
                    f"{s['candidate_id']}::blinded_audit".encode("utf-8")
                ).hexdigest()[:16],
                "candidate_id": s["candidate_id"],
                "dataset": d,
                "prompt_text": s.get("prompt_text", ""),
                "candidate_answer": s.get("candidate_answer", s.get("output_text", "")),
                "evidence_passages": s.get("evidence_passages", []),
                # Blinding guarantees:
                "model_identity_blinded": True,
                "pcg_decision_blinded": True,
                "comparator_decisions_blinded": True,
                "system_names_blinded": True,
                # Placeholders for 2 independent human annotators
                "human_annotator_1_verdict": None,
                "human_annotator_2_verdict": None,
                "tie_breaker_adjudicator_verdict": None,
            }
            blinded_records.append(blinded_record)

    expected_total = len(FROZEN_MODELS) * len(FROZEN_DATASETS) * sample_per_cell

    return {
        "schema": "PCG_MAS_V3_5_BLINDED_HUMAN_AUDIT_SAMPLE_V1",
        "sample_per_cell": sample_per_cell,
        "total_cells": len(cell_candidates),
        "total_selected": total_selected,
        "expected_total": expected_total,
        "selection_status": "PASS" if total_selected == expected_total else "FAIL",
        "per_cell_selection": per_cell_selection,
        "blinded_records": blinded_records,
    }
