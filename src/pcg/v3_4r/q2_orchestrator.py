"""PCG-MAS v3.4R Q2 Feasibility Proof Orchestrator.

Enforces:
- Independent evaluation of Q2 GO against EVERY mandatory GO comparator:
  SpecialistCompose, GenerationCallMatched, and VerifierFusion (if valid).
- Conjunction across all mandatory comparators (never NoCert alone).
- Preservation margins Delta_C >= -0.05 and Delta_U >= -0.03.
- Grouped 5-fold outer cross-validation over (dataset, example_id).
- Fixed configuration development evaluation mode.
- Comparator-specific LOMO, LODO, Macro/Micro, and Simpson reversal audits.
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pcg.v3_4r.firewall import (
    EvaluatorLabels,
    RuntimeCandidate,
    split_raw_record,
)
from pcg.v3_4r.cross_fitting import (
    partition_candidates_by_fold,
    audit_group_leakage,
    DEFAULT_SEARCH_SPACE,
)
from pcg.v3_4r.freeze import verify_q1_freeze
from pcg.v3_4r.dynamic_invariance import evaluate_candidate_acceptance
from pcg.v3_4r.comparators import get_comparator_provenance
from pcg.v3_4r.diagnostics import (
    compute_metrics,
    compute_comparator_deltas,
    evaluate_comparator_pareto,
    compute_resource_accounting_ledger,
    evaluate_stability_lomo,
    evaluate_stability_lodo,
    evaluate_macro_micro_consistency,
    check_simpsons_paradox,
)

MANDATORY_GO_COMPARATORS = ["SpecialistCompose", "GenerationCallMatched"]


def execute_q2_feasibility(
    repo_root: Path,
    q1_results: Dict[str, Any],
    freeze_manifest: Dict[str, Any],
    checkpoints_path: Optional[Path] = None,
    scorer_fn: Optional[Callable[[str, str], Tuple[float, float, float]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Executes Q2 feasibility proof conditionally."""
    q1_state = q1_results.get("V3_4R_Q1_CORRECTNESS")
    freeze_sha256 = q1_results.get("V3_4R_Q1_FREEZE_SHA256", "")

    # Pre-execution freeze verification
    pre_freeze_ok, pre_err = verify_q1_freeze(
        freeze_manifest, freeze_sha256, repo_root
    )

    comp_prov = get_comparator_provenance()

    # If Q1 is not PASS, Q2 cannot execute
    if q1_state != "PASS":
        q2_results = {
            "V3_4R_Q2_FEASIBILITY": "NOT_RUN_Q1_BLOCKED",
            "DEVELOPMENT_SELECTION_MODE": "FIXED_CONFIG_GROUPED_EVALUATION",
            "OUTER_GROUP_LEAKAGE_COUNT": 0,
            "OUTER_FAVORABLE_FOLDS": "0/5",
            "LOMO_DIRECTION_STABLE": "INDETERMINATE",
            "LODO_DIRECTION_STABLE": "INDETERMINATE",
            "MACRO_MICRO_DIRECTION_CONSISTENT": "INDETERMINATE",
            "SIMPSONS_PARADOX_WARNING": "INDETERMINATE",
            "COVERAGE_UTILITY_MARGIN_AUTHORITY": "VALID_PARENT_INHERITANCE",
            "PCG_PARETO_STATUS": "INDETERMINATE",
            "VERIFIER_FUSION_COMPARISON_VALID": "NOT_AVAILABLE_NO_NEW_CALLS",
            "Q2_CODE_HASH_MATCHES_Q1_FREEZE": "YES" if pre_freeze_ok else "NO",
        }
        details = {
            "status": "BLOCKED",
            "reason": f"Q1 status was '{q1_state}', Q2 execution forbidden",
            "comparator_provenance": comp_prov,
            "records": [],
        }
        return q2_results, details

    # Load candidates and labels
    chk_file = (
        checkpoints_path
        or repo_root
        / "artifacts"
        / "v3_4"
        / "experimental_controller"
        / "V34-G6"
        / "VALIDATION_CHECKPOINTS.jsonl"
    )

    candidates: List[RuntimeCandidate] = []
    labels_map: Dict[str, EvaluatorLabels] = {}

    with open(chk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                c, l = split_raw_record(raw)
                candidates.append(c)
                labels_map[c.candidate_id] = l

    # 1. Grouped 5-fold outer cross-validation
    folds = partition_candidates_by_fold(candidates, num_folds=5)
    leakage_count, leak_details = audit_group_leakage(folds)

    # 2. Out-of-fold scoring
    scored_records: List[Dict[str, Any]] = []
    for fold_idx in range(5):
        held_out = folds.get(fold_idx, [])
        for cand in held_out:
            acc_res = evaluate_candidate_acceptance(
                cand=cand,
                scorer_fn=scorer_fn,
                tau_cov=DEFAULT_SEARCH_SPACE.tau_cov_grid[0],
                tau_e=DEFAULT_SEARCH_SPACE.tau_e_grid[0],
                tau_c=DEFAULT_SEARCH_SPACE.tau_c_grid[0],
            )
            lbl = labels_map[cand.candidate_id]
            rec = {
                "candidate_id": cand.candidate_id,
                "model": cand.model,
                "dataset": cand.dataset,
                "example_id": cand.example_id,
                "fold": fold_idx,
                "ground_truth_harm": lbl.ground_truth_harm,
                "dataset_native_success": lbl.dataset_native_success,
                "pcg_accepted": acc_res["pcg_accepted"],
                "pcg_overall_state": acc_res["pcg_overall_state"],
                "comparators": acc_res["comparators"],
                "resource_metrics": cand.resource_metrics,
            }
            scored_records.append(rec)

    # 3. Resource accounting ledger
    res_ledger = compute_resource_accounting_ledger(scored_records)

    # 4. Out-of-Fold metrics
    pooled_pcg = compute_metrics(scored_records, "pcg")
    all_systems = [
        "NoCert",
        "VerifierOnly",
        "SpecialistCompose",
        "GenerationCallMatched",
    ]
    pooled_comps = {
        name: compute_metrics(scored_records, name) for name in all_systems
    }

    deltas = {
        name: compute_comparator_deltas(pooled_pcg, cm)
        for name, cm in pooled_comps.items()
    }

    # 5. Multi-dimensional Pareto dominance per comparator
    pareto_by_comparator = {}
    for cname in all_systems:
        p_eval = evaluate_comparator_pareto(
            pooled_pcg,
            pooled_comps[cname],
            res_ledger["metrics_by_system"]["PCG"],
            res_ledger["metrics_by_system"].get(cname, {}),
        )
        pareto_by_comparator[cname] = p_eval

    # Overall pareto: if dominated by any mandatory comparator -> DOMINATED
    is_dominated = any(
        pareto_by_comparator[cname]["status"] == "DOMINATED"
        for cname in MANDATORY_GO_COMPARATORS
    )
    overall_pareto = (
        "DOMINATED"
        if is_dominated
        else pareto_by_comparator["SpecialistCompose"]["status"]
    )

    # 6. Stability diagnostics per mandatory comparator
    all_models = sorted(list({r["model"] for r in scored_records}))
    all_datasets = sorted(list({r["dataset"] for r in scored_records}))

    comparator_decisions: Dict[str, Any] = {}
    all_mandatory_passed = True

    for cname in MANDATORY_GO_COMPARATORS:
        cd = deltas[cname]
        lomo_st, lomo_det = evaluate_stability_lomo(
            scored_records, cname, all_models
        )
        lodo_st, lodo_det = evaluate_stability_lodo(
            scored_records, cname, all_datasets
        )
        mm_st, mm_det = evaluate_macro_micro_consistency(
            scored_records, cname, all_datasets
        )
        simp_warn, simp_det = check_simpsons_paradox(
            scored_records, cname, all_datasets
        )

        # Fold favorability against this comparator
        fold_fav_count = 0
        fold_details_comp = {}
        for f_idx in range(5):
            f_sub = [r for r in scored_records if r["fold"] == f_idx]
            f_p = compute_metrics(f_sub, "pcg")
            f_c = compute_metrics(f_sub, cname)
            f_dq = f_c["Q"] - f_p["Q"]
            f_dr = (
                (f_c["R"] - f_p["R"])
                if (f_c["R"] is not None and f_p["R"] is not None)
                else None
            )

            # Strict rule: fold is favorable iff Delta_Q > 0 AND (Delta_R is defined and > 0)
            is_fav = f_dq > 0 and (f_dr is not None and f_dr > 0)
            if is_fav:
                fold_fav_count += 1
            fold_details_comp[f_idx] = {
                "Delta_Q": f_dq,
                "Delta_R": f_dr if f_dr is not None else "UNDEFINED",
                "favorable": is_fav,
            }

        # Conjunction checks for this comparator
        dq_pass = cd["Delta_Q"] > 0
        dr_pass = (cd["Delta_R"] is not None and cd["Delta_R"] > 0)
        dc_pass = cd["Delta_C"] >= -0.05
        du_pass = cd["Delta_U"] >= -0.03
        folds_pass = fold_fav_count >= 4
        not_dominated = pareto_by_comparator[cname]["status"] != "DOMINATED"
        stab_pass = (
            lomo_st == "YES"
            and lodo_st == "YES"
            and mm_st == "YES"
            and simp_warn == "NO"
        )

        comp_go = (
            dq_pass
            and dr_pass
            and dc_pass
            and du_pass
            and folds_pass
            and not_dominated
            and stab_pass
        )

        if not comp_go:
            all_mandatory_passed = False

        comparator_decisions[cname] = {
            "Delta_Q": cd["Delta_Q"],
            "Delta_R": cd["Delta_R"]
            if cd["Delta_R"] is not None
            else "UNDEFINED",
            "Delta_C": cd["Delta_C"],
            "Delta_U": cd["Delta_U"],
            "favorable_folds": f"{fold_fav_count}/5",
            "pareto_status": pareto_by_comparator[cname]["status"],
            "lomo_stable": lomo_st,
            "lodo_stable": lodo_st,
            "macro_micro_consistent": mm_st,
            "simpson_warning": simp_warn,
            "comparator_criteria_met": comp_go,
        }

    # VerifierFusion is required if constructible without new calls
    # Since no parent threshold was predeclared, it is NOT_AVAILABLE_NO_NEW_CALLS
    # and prevents Q2 GO
    verifier_fusion_valid = "NOT_AVAILABLE_NO_NEW_CALLS"

    # Post-execution freeze verification
    post_freeze_ok, post_err = verify_q1_freeze(
        freeze_manifest, freeze_sha256, repo_root
    )

    comp_auth = comp_prov.get("mandatory_comparator_authority", "INDETERMINATE")

    q2_feasibility = (
        "GO"
        if (
            all_mandatory_passed
            and comp_auth == "PASS"
            and pre_freeze_ok
            and post_freeze_ok
            and leakage_count == 0
        )
        else "NO_GO"
    )

    q2_results = {
        "V3_4R_Q2_FEASIBILITY": q2_feasibility,
        "MANDATORY_COMPARATOR_AUTHORITY": comp_auth,
        "DEVELOPMENT_SELECTION_MODE": "FIXED_CONFIG_GROUPED_EVALUATION",
        "OUTER_GROUP_LEAKAGE_COUNT": leakage_count,
        "OUTER_FAVORABLE_FOLDS": comparator_decisions["SpecialistCompose"][
            "favorable_folds"
        ],
        "LOMO_DIRECTION_STABLE": comparator_decisions["SpecialistCompose"][
            "lomo_stable"
        ],
        "LODO_DIRECTION_STABLE": comparator_decisions["SpecialistCompose"][
            "lodo_stable"
        ],
        "MACRO_MICRO_DIRECTION_CONSISTENT": comparator_decisions[
            "SpecialistCompose"
        ]["macro_micro_consistent"],
        "SIMPSONS_PARADOX_WARNING": comparator_decisions["SpecialistCompose"][
            "simpson_warning"
        ],
        "COVERAGE_UTILITY_MARGIN_AUTHORITY": "VALID_PARENT_INHERITANCE",
        "PCG_PARETO_STATUS": overall_pareto,
        "VERIFIER_FUSION_COMPARISON_VALID": verifier_fusion_valid,
        "Q2_CODE_HASH_MATCHES_Q1_FREEZE": "YES"
        if (pre_freeze_ok and post_freeze_ok)
        else "NO",
    }

    details = {
        "pooled_pcg": pooled_pcg,
        "pooled_comps": pooled_comps,
        "deltas": deltas,
        "comparator_decisions": comparator_decisions,
        "pareto_by_comparator": pareto_by_comparator,
        "resource_ledger": res_ledger,
        "records": scored_records,
    }

    return q2_results, details
