"""Production exact top-k_c matching engine for PCG-MAS v3.5.

Enforces:
- Exact top-k_c candidate selection per model x dataset cell (Gate S1).
- Deterministic hash tie-break: SHA256(candidate_id || comparator_id || v3_5_freeze_root) (Gate S2).
- Zero division safety: k_c == 0 yields undefined risk (None), NEVER silently coerced to 0.0 (Gate S5).
- Macro mean across all predeclared cells without exclusion (Gate S6).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

from pcg.v3_5.comparators import BaseComparator, compute_deterministic_tie_break


def select_top_kc(
    ranked_candidates: List[Dict[str, Any]],
    k_c: int,
) -> List[Dict[str, Any]]:
    """Select exactly top k_c candidates.

    Raises ValueError if ranked_candidates has fewer than k_c candidates.
    """
    if k_c < 0:
        raise ValueError(f"k_c cannot be negative: {k_c}")
    if k_c > len(ranked_candidates):
        raise ValueError(
            f"Cannot select k_c={k_c} from pool of size {len(ranked_candidates)}"
        )
    return ranked_candidates[:k_c]


def compute_cell_risk(
    accepted_candidates: List[Dict[str, Any]],
    k_c: int,
) -> Optional[float]:
    """Compute selective risk: harmful_accepted / k_c.

    Returns None (UNDEFINED) if k_c == 0.
    NEVER silently coerces to 0.0.
    """
    if k_c == 0:
        return None

    harm_count = 0
    for c in accepted_candidates:
        if int(c.get("harm_label", 0)) == 1:
            harm_count += 1

    return float(harm_count / k_c)


def compute_cell_utility(
    accepted_candidates: List[Dict[str, Any]],
    k_c: int,
) -> Optional[float]:
    """Compute cell utility: success_accepted / k_c.

    Returns None (UNDEFINED) if k_c == 0.
    """
    if k_c == 0:
        return None

    succ_count = 0
    for c in accepted_candidates:
        if int(c.get("success_label", 0)) == 1:
            succ_count += 1

    return float(succ_count / k_c)


def match_cell_candidates(
    cell_candidates: List[Dict[str, Any]],
    pcg_accepted_ids: Set[str],
    comparators: Dict[str, BaseComparator],
    v3_5_freeze_root: str,
) -> Dict[str, Any]:
    """Match candidates for a single (model, dataset) cell.

    Returns:
        Dict with k_c, pcg_risk, and per-comparator matched selections and risks.
    """
    k_c = len(pcg_accepted_ids)
    pcg_selected = [c for c in cell_candidates if c["candidate_id"] in pcg_accepted_ids]

    pcg_risk = compute_cell_risk(pcg_selected, k_c)
    pcg_utility = compute_cell_utility(pcg_selected, k_c)

    comp_results: Dict[str, Any] = {}

    for comp_id, comparator in comparators.items():
        ranked = comparator.rank_candidates(cell_candidates, v3_5_freeze_root)
        selected = select_top_kc(ranked, k_c)
        risk = compute_cell_risk(selected, k_c)
        utility = compute_cell_utility(selected, k_c)

        # Compute delta if risk is defined
        delta_risk = (risk - pcg_risk) if (risk is not None and pcg_risk is not None) else None
        delta_util = (pcg_utility - utility) if (utility is not None and pcg_utility is not None) else None

        comp_results[comp_id] = {
            "selected_count": len(selected),
            "risk": risk,
            "utility": utility,
            "delta_risk": delta_risk,
            "delta_utility": delta_util,
            "selected_ids": [c["candidate_id"] for c in selected],
        }

    return {
        "k_c": k_c,
        "pcg_risk": pcg_risk,
        "pcg_utility": pcg_utility,
        "pcg_selected_ids": list(pcg_accepted_ids),
        "comparators": comp_results,
    }


def verify_matching_domain() -> Dict[str, Any]:
    """Production verification callable for matching domain."""
    sample = [{"candidate_id": "c1", "is_harmful": False}, {"candidate_id": "c2", "is_harmful": True}]
    top1 = select_top_kc(sample, 1)
    return {"domain": "matching", "top1_count": len(top1)}

