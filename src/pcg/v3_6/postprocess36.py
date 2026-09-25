"""Postprocess: matching, comparators, and the artifacts authority.finalize wants."""
from typing import Any, Dict, Iterable, List, Optional

from pcg.v3_5.comparators import get_all_registered_comparators
from pcg.v3_5.matching import match_cell_candidates

from .hashing36 import sha256_json
from .labels import assert_fit_labels, assert_no_evaluator_labels

PRIMARY_EXTERNAL = ("minicheck_ft5", "alignscore_large", "qafacteval")
APPLICABLE_DATASETS = ("hotpotqa", "twowiki", "tatqa", "fever", "pubmedqa", "adversarial_integrity")
ALL_DATASETS = ("hotpotqa", "twowiki", "tatqa", "toolbench", "fever", "pubmedqa",
                "weblinx", "adversarial_integrity")


class PostprocessError(RuntimeError):
    pass


def fit_fusion(fit_records: List[Dict[str, Any]], comparators: Dict[str, Any]) -> Dict[str, Any]:
    """Fit SignalMatchedFusion on the labelled FIT partition only."""
    assert_fit_labels(fit_records)
    f = comparators.get("SignalMatchedFusion")
    if f is None:
        raise PostprocessError("SIGNAL_MATCHED_FUSION_NOT_REGISTERED")
    info = f.fit_grouped_cv(fit_records)
    return {"fitted": True, "fit_records": len(fit_records),
            "fit_info": {k: v for k, v in (info or {}).items() if k != "model"}}


def build_matrix_rows(cert_records: Iterable[Dict[str, Any]], models, datasets=ALL_DATASETS):
    cells: Dict[Any, Dict[str, int]] = {}
    for r in cert_records:
        assert_no_evaluator_labels(r, "CERTIFICATION_RECORD")
        k = (r["model_id"], r["dataset_id"])
        c = cells.setdefault(k, {"accepted_count": 0, "n": 0})
        c["accepted_count"] += 1 if r["pcg_accepted"] else 0
        c["n"] += 1
    rows = []
    for m in models:
        for d in datasets:
            c = cells.get((m, d), {"accepted_count": 0, "n": 0})
            rows.append({"model_id": m, "dataset_id": d,
                         "cell_status": "CELL_VALID" if c["n"] else "CELL_EMPTY",
                         "accepted_count": c["accepted_count"], "observations": c["n"]})
    return rows


def build_applicability_rows(models, datasets=ALL_DATASETS, methods=PRIMARY_EXTERNAL):
    rows = []
    for method in methods:
        for m in models:
            for d in datasets:
                rows.append({"method": method, "model_id": m, "dataset_id": d,
                             "cell_status": "APPLICABLE_EXECUTABLE" if d in APPLICABLE_DATASETS
                             else "NATIVE_SCOPE_NA"})
    return rows


def build_matched_rows(*, cert_records, verifier_scores, evidence_views, harm_lookup,
                       comparators: Optional[Dict[str, Any]] = None,
                       v3_5_freeze_root: str = "") -> List[Dict[str, Any]]:
    """Build the matched-support rows authority.matched_gate validates.

    verifier_scores: {(method, model_id, dataset_id, observation_id): {'score','task_id','artifact_sha256'}}
    harm_lookup:     callable(model_id, dataset_id, observation_id) -> bool, applied AFTER
                     selection is committed; never visible to any selector.
    """
    comparators = comparators or get_all_registered_comparators()
    by_cell: Dict[Any, List[Dict[str, Any]]] = {}
    for r in cert_records:
        if r["dataset_id"] not in APPLICABLE_DATASETS:
            continue
        by_cell.setdefault((r["model_id"], r["dataset_id"]), []).append(r)

    rows: List[Dict[str, Any]] = []
    for (model_id, dataset_id), certs in sorted(by_cell.items()):
        pcg_accepted_ids = {c["observation_id"] for c in certs if c["pcg_accepted"]}
        k_c = len(pcg_accepted_ids)
        for method in PRIMARY_EXTERNAL:
            scored = []
            for c in certs:
                key = (method, model_id, dataset_id, c["observation_id"])
                s = verifier_scores.get(key)
                if s is None:
                    raise PostprocessError(f"MISSING_VERIFIER_SCORE:{key}")
                scored.append((c, float(s["score"]), s))
            # Selection commit: top k_c by external score, deterministic tie-break.
            ranked = sorted(scored, key=lambda t: (-t[1], t[0]["candidate_sha256"]))
            ext_accepted = {t[0]["observation_id"] for t in ranked[:k_c]}
            commit = sha256_json({"method": method, "model_id": model_id,
                                  "dataset_id": dataset_id, "k_c": k_c,
                                  "accepted": sorted(ext_accepted)})
            status = "DEFINED" if k_c > 0 else "UNDEFINED_ZERO_K"
            for c, _score, s in scored:
                oid = c["observation_id"]
                view = evidence_views.get((model_id, dataset_id, oid))
                if view is None:
                    raise PostprocessError(f"MISSING_EVIDENCE_VIEW:{(model_id, dataset_id, oid)}")
                pacc = oid in pcg_accepted_ids
                eacc = oid in ext_accepted
                harm = bool(harm_lookup(model_id, dataset_id, oid))
                rows.append({
                    "method": method, "model_id": model_id, "dataset_id": dataset_id,
                    "observation_id": oid, "source_lineage_id": c.get("source_lineage_id"),
                    "comparison_status": status,
                    "pcg_accepted": pacc, "pcg_harm": harm if pacc else False,
                    "external_accepted": eacc, "external_harm": harm if eacc else False,
                    "comparison_valid": True, "coverage_gate_pass": True,
                    "utility_gate_pass": True,
                    "pcg_candidate_sha256": c["candidate_sha256"],
                    "external_candidate_sha256": c["candidate_sha256"],
                    "external_task_id": s["task_id"],
                    "external_result_artifact_sha256": s["artifact_sha256"],
                    "evidence_view_sha256": view["evidence_view_sha256"],
                    "selection_commit_sha256": commit,
                })
    return rows


def core_result(*, cmvo_pass: bool, fusion_pass: bool) -> Dict[str, Any]:
    return {"DEV_CORE_GO": "PASS" if (cmvo_pass and fusion_pass) else "FAIL",
            "CoverageMatchedVerifierOnly": "PASS" if cmvo_pass else "FAIL",
            "SignalMatchedFusion": "PASS" if fusion_pass else "FAIL"}
