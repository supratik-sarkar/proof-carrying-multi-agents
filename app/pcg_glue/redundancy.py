"""
Demo-runtime implementation of the PCG-MAS paper algorithm.

This module implements the same public semantics as the redundancy selector
described in the paper appendix (Template — "Redundancy selector"):

    Chooses k certificates satisfying provenance, tool-overlap, and replayable-
    overlap separation. Produces selected certificate IDs and pairwise
    overlap scores.

It is intentionally written against the lightweight demo schema in
app/pcg_glue/schemas.py so the Hugging Face Space remains self-contained.

For the demo, "independent claims" means: pairs of ClaimCertificates whose
support_ids do not overlap (provenance independence) and whose tool_output_ids
do not overlap (tool independence). The selector greedily picks the top-k
accepted claims maximizing minimum pairwise independence.

This is the demo-runtime instantiation of the R2 redundancy experiment
(scripts/experiments/run_r2_redundancy.py uses the full estimator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pcg_glue.schemas import ClaimCertificate


# =============================================================================
# Pairwise overlap between two claim certificates
# =============================================================================

def overlap_score(a: ClaimCertificate, b: ClaimCertificate) -> float:
    """Jaccard overlap between (support_ids ∪ tool_output_ids) of two claims.

    Returns 0.0 for fully independent claims, 1.0 for identical support.
    """
    s_a = set(a.claim.support_ids) | {f"tool:{t}" for t in a.claim.tool_output_ids}
    s_b = set(b.claim.support_ids) | {f"tool:{t}" for t in b.claim.tool_output_ids}
    if not s_a and not s_b:
        return 0.0
    inter = s_a & s_b
    union = s_a | s_b
    return len(inter) / len(union) if union else 0.0


# =============================================================================
# Redundancy selector
# =============================================================================

@dataclass
class RedundancySelection:
    """Result of selecting k independent claim certificates."""
    selected_ids: list[str]
    pairwise_overlap: dict[tuple[str, str], float]
    k: int
    n_accepted_candidates: int

    def to_dict(self) -> dict:
        return {
            "selected_ids": self.selected_ids,
            "pairwise_overlap": {f"{a}|{b}": v
                                 for (a, b), v in self.pairwise_overlap.items()},
            "k": self.k,
            "n_accepted_candidates": self.n_accepted_candidates,
        }


def select_independent(
    claim_certs: list[ClaimCertificate],
    k: int = 3,
    overlap_threshold: float = 0.5,
) -> RedundancySelection:
    """Greedy selection of up to k accepted claim-certificates with low overlap.

    Algorithm:
        1. Filter to accepted claims only.
        2. Sort by confidence descending (highest-confidence first).
        3. Greedily pick claims whose pairwise overlap vs already-selected is
           below `overlap_threshold`.
        4. Stop when k claims are selected or candidates are exhausted.
        5. Compute pairwise overlap on the final selection for the report.

    Returns a RedundancySelection. If fewer than k claims survive, the
    selection contains what was achievable.
    """
    accepted = [cc for cc in claim_certs if cc.accepted]
    accepted.sort(key=lambda cc: cc.claim.confidence, reverse=True)

    selected: list[ClaimCertificate] = []
    for cc in accepted:
        if len(selected) >= k:
            break
        if all(overlap_score(cc, s) < overlap_threshold for s in selected):
            selected.append(cc)

    selected_ids = [s.claim.claim_id for s in selected]
    pairwise: dict[tuple[str, str], float] = {}
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            pairwise[(a.claim.claim_id, b.claim.claim_id)] = round(
                overlap_score(a, b), 3,
            )

    return RedundancySelection(
        selected_ids=selected_ids,
        pairwise_overlap=pairwise,
        k=len(selected),
        n_accepted_candidates=len(accepted),
    )


__all__ = [
    "select_independent",
    "overlap_score",
    "RedundancySelection",
]
