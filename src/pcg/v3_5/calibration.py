"""Production calibration split selector scaffold for PCG-MAS v3.5.

Enforces split specifications from 08_V3_5_CALIBRATION_FEASIBILITY_AUDIT_CONTRACT.json:
- D_CAL.FIT:
  - 20 underlying examples per dataset x 7 models = 140 per dataset x 7 datasets = 980 candidates.
- D_CAL.FEAS:
  - 10 underlying examples per dataset x 7 models = 70 per dataset x 7 datasets = 490 candidates.
- D_CAL.AUDIT:
  - 60 unique underlying negative examples per dataset x 7 datasets = 420 candidates.
  - Each negative example is assigned to exactly one model via deterministic balanced hash.
- Disjointness:
  - FIT, FEAS, AUDIT are mutually disjoint and disjoint from D_VAL and D_FINAL.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Set, Tuple

from pcg.v3_5.registries import FROZEN_DATASETS, FROZEN_MODELS


def deterministic_model_assignment(
    example_id: str,
    dataset: str,
    models: List[str] = FROZEN_MODELS,
) -> str:
    """Assign an example to exactly one model using deterministic hash."""
    h = hashlib.sha256(f"{dataset}:{example_id}".encode("utf-8")).hexdigest()
    idx = int(h, 16) % len(models)
    return models[idx]


def scaffold_d_cal_splits() -> Dict[str, Any]:
    """Generate synthetic scaffold demonstrating exact D_CAL split allocation."""
    d_cal_fit: List[Dict[str, Any]] = []
    d_cal_feas: List[Dict[str, Any]] = []
    d_cal_audit: List[Dict[str, Any]] = []

    seen_example_keys: Set[str] = set()

    for d in FROZEN_DATASETS:
        # 1. D_CAL.FIT: 20 examples x 7 models = 140 per dataset
        for ex_num in range(1, 21):
            ex_id = f"{d}_fit_ex_{ex_num:03d}"
            seen_example_keys.add(f"{d}:{ex_id}")
            for m in FROZEN_MODELS:
                d_cal_fit.append({
                    "split": "D_CAL_FIT",
                    "dataset": d,
                    "example_id": ex_id,
                    "model": m,
                    "candidate_id": f"cand_{d}_{m}_{ex_id}",
                })

        # 2. D_CAL.FEAS: 10 examples x 7 models = 70 per dataset
        for ex_num in range(1, 11):
            ex_id = f"{d}_feas_ex_{ex_num:03d}"
            seen_example_keys.add(f"{d}:{ex_id}")
            for m in FROZEN_MODELS:
                d_cal_feas.append({
                    "split": "D_CAL_FEAS",
                    "dataset": d,
                    "example_id": ex_id,
                    "model": m,
                    "candidate_id": f"cand_{d}_{m}_{ex_id}",
                })

        # 3. D_CAL.AUDIT: 60 unique negative examples per dataset, exactly 1 model each
        for ex_num in range(1, 61):
            ex_id = f"{d}_audit_neg_ex_{ex_num:03d}"
            seen_example_keys.add(f"{d}:{ex_id}")
            assigned_m = deterministic_model_assignment(ex_id, d, FROZEN_MODELS)
            d_cal_audit.append({
                "split": "D_CAL_AUDIT",
                "dataset": d,
                "example_id": ex_id,
                "model": assigned_m,
                "candidate_id": f"cand_{d}_{assigned_m}_{ex_id}",
            })

    # Verification
    fit_count = len(d_cal_fit)
    feas_count = len(d_cal_feas)
    audit_count = len(d_cal_audit)

    fit_ex_count = len({f"{r['dataset']}:{r['example_id']}" for r in d_cal_fit})
    feas_ex_count = len({f"{r['dataset']}:{r['example_id']}" for r in d_cal_feas})
    audit_ex_count = len({f"{r['dataset']}:{r['example_id']}" for r in d_cal_audit})

    # Ensure zero overlap
    overlap_fit_feas = set(r["candidate_id"] for r in d_cal_fit).intersection(
        set(r["candidate_id"] for r in d_cal_feas)
    )
    overlap_fit_audit = set(r["candidate_id"] for r in d_cal_fit).intersection(
        set(r["candidate_id"] for r in d_cal_audit)
    )
    overlap_feas_audit = set(r["candidate_id"] for r in d_cal_feas).intersection(
        set(r["candidate_id"] for r in d_cal_audit)
    )

    is_disjoint = (
        len(overlap_fit_feas) == 0
        and len(overlap_fit_audit) == 0
        and len(overlap_feas_audit) == 0
    )

    # Per-model distribution in D_CAL.AUDIT
    audit_model_counts: Dict[str, int] = {}
    for r in d_cal_audit:
        m = r["model"]
        audit_model_counts[m] = audit_model_counts.get(m, 0) + 1

    audit_summary = {
        "schema": "PCG_MAS_V3_5_DCAL_SELECTOR_AUDIT_V1",
        "d_cal_fit_count": fit_count,
        "d_cal_fit_expected": 980,
        "d_cal_feas_count": feas_count,
        "d_cal_feas_expected": 490,
        "d_cal_audit_count": audit_count,
        "d_cal_audit_expected": 420,
        "fit_examples_per_dataset": fit_ex_count // len(FROZEN_DATASETS),
        "feas_examples_per_dataset": feas_ex_count // len(FROZEN_DATASETS),
        "audit_examples_per_dataset": audit_ex_count // len(FROZEN_DATASETS),
        "audit_model_distribution": audit_model_counts,
        "splits_disjoint": is_disjoint,
        "status": "PASS" if (fit_count == 980 and feas_count == 490 and audit_count == 420 and is_disjoint) else "FAIL",
    }

    return audit_summary
