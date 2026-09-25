"""DEV-only adversarial sampling amendment.

Prospective, dated, content-addressed. Selects a deterministic subset of the
frozen FINAL adversarial pairs before benchmark generation #1. The FULL_FINAL
697-pair population remains the PROD authority and is not redefined here.
"""
import hashlib
from datetime import date
from typing import Any, Dict, Iterable, List

from .hashing36 import sha256_json

AMENDMENT_ID = "PCG_MAS_V3_6_DEV_ADV_AMENDMENT_2026-09-14"
AMENDMENT_DATE = "2026-09-14"
DEV_SELECTED_ADV_PAIRS = 300
ITEMS_PER_PAIR = 2
DEV_SELECTED_ADV_ITEMS = DEV_SELECTED_ADV_PAIRS * ITEMS_PER_PAIR
FULL_FINAL_PAIRS = 697
SCOPE = "DEV_ONLY"


class AmendmentError(RuntimeError):
    pass


def _rank_key(pair_id: str, amendment_id: str = AMENDMENT_ID) -> str:
    return hashlib.sha256(f"{amendment_id}|{pair_id}".encode("utf-8")).hexdigest()


def select_pairs(all_pair_ids: Iterable[str], n: int = DEV_SELECTED_ADV_PAIRS,
                 amendment_id: str = AMENDMENT_ID) -> List[str]:
    ids = sorted({str(p) for p in all_pair_ids})
    if len(ids) < n:
        raise AmendmentError(f"INSUFFICIENT_FINAL_PAIRS:have={len(ids)}:need={n}")
    return sorted(ids, key=lambda p: (_rank_key(p, amendment_id), p))[:n]


def build_manifest(all_pair_ids: Iterable[str], item_ids_for_pair,
                   n: int = DEV_SELECTED_ADV_PAIRS) -> Dict[str, Any]:
    """item_ids_for_pair: callable(pair_id) -> sequence of exactly 2 item ids."""
    ids = sorted({str(p) for p in all_pair_ids})
    if len(ids) != FULL_FINAL_PAIRS:
        raise AmendmentError(f"FINAL_POPULATION_NOT_697:got={len(ids)}")
    chosen = select_pairs(ids, n)
    items: List[str] = []
    for p in chosen:
        got = [str(x) for x in item_ids_for_pair(p)]
        if len(got) != ITEMS_PER_PAIR:
            raise AmendmentError(f"PAIR_NOT_COMPLETE:{p}:items={len(got)}")
        items.extend(got)
    if len(items) != n * ITEMS_PER_PAIR:
        raise AmendmentError("ITEM_CARDINALITY_MISMATCH")
    core = {
        "schema": "PCG_MAS_V3_6_DEV_ADV_AMENDMENT_MANIFEST_V1",
        "amendment_id": AMENDMENT_ID, "amendment_date": AMENDMENT_DATE, "scope": SCOPE,
        "rationale": ("Equal-weight macro over model x dataset cells gives the adversarial "
                      "cell one eighth of a row regardless of its observation count. The "
                      "FULL_FINAL population spends the majority of the execution budget on "
                      "per-cell precision the estimand cannot use. Declared before benchmark "
                      "generation #1; no DEV result observed."),
        "selection_algorithm": "sort by SHA256(amendment_id|pair_id) ascending, take first N, keep both pair members",
        "full_final_pairs": FULL_FINAL_PAIRS,
        "dev_selected_adv_pairs": len(chosen),
        "dev_selected_adv_items": len(items),
        "selected_pair_ids": chosen,
        "selected_item_ids": sorted(items),
        "prod_authority_unchanged": True,
        "post_result_reselection": "FORBIDDEN",
    }
    return {**core, "manifest_sha256": sha256_json(core)}


def verify_manifest(manifest: Dict[str, Any]) -> bool:
    core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if sha256_json(core) != manifest.get("manifest_sha256"):
        raise AmendmentError("AMENDMENT_MANIFEST_HASH_MISMATCH")
    if manifest["dev_selected_adv_pairs"] != DEV_SELECTED_ADV_PAIRS:
        raise AmendmentError("AMENDMENT_PAIR_COUNT")
    if manifest["dev_selected_adv_items"] != DEV_SELECTED_ADV_ITEMS:
        raise AmendmentError("AMENDMENT_ITEM_COUNT")
    if len(set(manifest["selected_pair_ids"])) != DEV_SELECTED_ADV_PAIRS:
        raise AmendmentError("AMENDMENT_PAIR_DUPLICATES")
    return True


def dev_cardinalities(natural_obs_per_model: int = 280, models: int = 7,
                      verifiers: int = 3, applicable_natural_datasets: int = 5,
                      natural_obs_per_dataset: int = 40) -> Dict[str, int]:
    obs = natural_obs_per_model + DEV_SELECTED_ADV_ITEMS
    per_v = applicable_natural_datasets * natural_obs_per_dataset + DEV_SELECTED_ADV_ITEMS
    return {
        "observations_per_model": obs,
        "total_generation_tasks": obs * models,
        "external_scores_per_verifier_per_model": per_v,
        "external_expected_numerical_scores": per_v * models * verifiers,
        "dev_selected_adv_pairs": DEV_SELECTED_ADV_PAIRS,
        "dev_selected_adv_items": DEV_SELECTED_ADV_ITEMS,
    }
