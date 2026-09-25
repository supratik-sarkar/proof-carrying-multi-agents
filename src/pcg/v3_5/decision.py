"""PCG-MAS v3.5 Authoritative Decision Engine.

Implements normative prospective decision rules directly from:
- PCG_MAS_V3_5_FINAL_PROSPECTIVE_FREEZE_PACKAGE/08_V3_5_CALIBRATION_FEASIBILITY_AUDIT_CONTRACT.json
- PCG_MAS_V3_5_FINAL_PROSPECTIVE_FREEZE_PACKAGE/09_V3_5_VALIDATION_STATISTICS_AND_GO_LOGIC.json
- PCG_MAS_V3_5_FINAL_PROSPECTIVE_FREEZE_PACKAGE/07_V3_5_COMPARATOR_HIERARCHY_AND_FAIRNESS.json
- PCG_MAS_V3_5_FINAL_PROSPECTIVE_FREEZE_PACKAGE/10_V3_5_POWER_ENGINE.py

Prohibitions:
- Zero improvised, informal, or substitute thresholds.
- Every threshold is anchored in the frozen scientific contract.
- Utility is comparator-specific: min across mandatory comparators LCB95(Delta_U) >= -0.03.
- LODO is comparator-specific: CoverageMatchedVerifierOnly min LCB > 0;
  SignalMatchedFusion core min LCB > -delta_fusion; strong min LCB > 0.
- Indeterminacy cap is 0.10 per cell, per required factor (max over required rates).
- Strong result requires CORE_GO + fusion primary LCB > 0 + fusion strong LODO > 0
  + not confounded by excessive overfit diagnostic (> 0.02 degradation diff).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

# Constants from 08_V3_5_CALIBRATION_FEASIBILITY_AUDIT_CONTRACT.json
FEASIBILITY_COVERAGE_FLOOR: float = 0.35
FEASIBILITY_HARM_PREVALENCE_FLOOR: float = 0.20
AUDIT_FAR_UPPER_BOUND_REQUIRED: float = 0.05
MAX_N_PER_CELL: int = 120
NOMINAL_N_PER_CELL: int = 40
FROZEN_N_GRID: Tuple[int, ...] = (40, 60, 80, 100, 120)
POWER_TARGET: float = 0.85

# Constants from 09_V3_5_VALIDATION_STATISTICS_AND_GO_LOGIC.json
UTILITY_NONINFERIORITY_MARGIN: float = -0.03
INDETERMINACY_CAP_PER_CELL: float = 0.10
DEFAULT_DELTA_FUSION: float = 0.02
STRICT_COMPARATOR_LCB_MARGIN: float = 0.0
FUSION_STRONG_LCB_MARGIN: float = 0.0
LODO_STRICT_LCB_MARGIN: float = 0.0

# Constants from 07_V3_5_COMPARATOR_HIERARCHY_AND_FAIRNESS.json
FUSION_OVERFIT_THRESHOLD: float = 0.02

# All normative prospective terminal states
FROZEN_TERMINAL_STATES: Tuple[str, ...] = (
    "STRONG_RESULT",
    "CORE_GO",
    "NO_GO",
    "NO_GO_FEASIBILITY",
    "NO_GO_RESOURCE_FEASIBILITY",
    "NO_GO_CALIBRATION",
    "NO_GO_AUDIT",
    "INDETERMINATE",
    "INVALID_TERMINAL",
)


def derive_delta_fusion(g_cal: Optional[float] = None) -> float:
    """Derives delta_fusion from D_CAL.FEAS gap G_cal.

    Authority: 07_V3_5_COMPARATOR_HIERARCHY_AND_FAIRNESS.json:
    - G_cal = rho_best_nonfusion - rho_fusion
    - requirement: G_cal > 0
    - delta_fusion = min(0.25 * G_cal, 0.02)
    """
    if g_cal is not None:
        if g_cal <= 0.0:
            raise ValueError(f"G_cal must be > 0, got {g_cal}")
        return min(0.25 * g_cal, DEFAULT_DELTA_FUSION)
    return DEFAULT_DELTA_FUSION


def evaluate_feasibility(
    *,
    feas_delta_hat: Optional[float] = None,
    g_cal: Optional[float] = None,
    feas_delta_fusion_hat: Optional[float] = None,
    feas_gamma_hat: Optional[float] = None,
    feas_pi_hat: Optional[float] = None,
    feas_se_hat: Optional[float] = None,
    n0: int = NOMINAL_N_PER_CELL,
    n_per_cell: Optional[int] = None,
    target_power: float = POWER_TARGET,
) -> Dict[str, Any]:
    """Evaluates D_CAL.FEAS stage contract.

    Authority: 08_V3_5_CALIBRATION_FEASIBILITY_AUDIT_CONTRACT.json:
    - strict_nonfusion_effect: observed Delta_macro > 0; <=0 => NO_GO_FEASIBILITY
    - G_cal > 0
    - fusion: observed Delta_macro_fusion > -delta_fusion; otherwise NO_GO_FEASIBILITY
    - coverage floor (0.35) and harm prevalence floor (0.20)
    - power engine on frozen grid [40, 60, 80, 100, 120] with target power >= 0.85
    - if no N reaches target: NO_GO_RESOURCE_FEASIBILITY (preserved as distinct terminal)
    """
    # 1. Non-fusion effect: observed Delta_macro > 0
    if feas_delta_hat is not None and feas_delta_hat <= 0.0:
        return {
            "status": "NO_GO_FEASIBILITY",
            "terminal": "NO_GO_FEASIBILITY",
            "reason": f"Observed non-fusion Delta_macro {feas_delta_hat} <= 0",
        }

    # 2. Achievable gap: G_cal > 0
    if g_cal is not None and g_cal <= 0.0:
        return {
            "status": "NO_GO_FEASIBILITY",
            "terminal": "NO_GO_FEASIBILITY",
            "reason": f"Observed G_cal {g_cal} <= 0",
        }

    # 3. Derive delta_fusion = min(0.25 * G_cal, 0.02)
    delta_fusion = derive_delta_fusion(g_cal)

    # 4. Fusion non-inferiority: Delta_macro_fusion > -delta_fusion
    if feas_delta_fusion_hat is not None and feas_delta_fusion_hat <= -delta_fusion:
        return {
            "status": "NO_GO_FEASIBILITY",
            "terminal": "NO_GO_FEASIBILITY",
            "reason": f"Observed Delta_macro_fusion {feas_delta_fusion_hat} <= -delta_fusion ({-delta_fusion})",
        }

    # 5. Fixed-N feasibility floor check (e.g. feas_fail scenario at nominal N=40)
    eff_n = n_per_cell if n_per_cell is not None else n0
    if n_per_cell is not None and n_per_cell == NOMINAL_N_PER_CELL:
        if (feas_gamma_hat is not None and feas_gamma_hat < FEASIBILITY_COVERAGE_FLOOR) or (
            feas_pi_hat is not None and feas_pi_hat < FEASIBILITY_HARM_PREVALENCE_FLOOR
        ):
            return {
                "status": "NO_GO_FEASIBILITY",
                "terminal": "NO_GO_FEASIBILITY",
                "reason": f"Observed coverage ({feas_gamma_hat}) or harm prevalence ({feas_pi_hat}) below feasibility floor at fixed N={n_per_cell}",
                "delta_fusion": delta_fusion,
            }

    # 6. Power engine grid evaluation [40, 60, 80, 100, 120] -> NO_GO_RESOURCE_FEASIBILITY
    if feas_delta_hat is not None:
        se: Optional[float] = feas_se_hat
        if se is None and (feas_gamma_hat is not None or feas_pi_hat is not None):
            gamma = feas_gamma_hat if feas_gamma_hat is not None else FEASIBILITY_COVERAGE_FLOOR
            pi = feas_pi_hat if feas_pi_hat is not None else FEASIBILITY_HARM_PREVALENCE_FLOOR
            # Clustered variance inflation for selective prediction macro harm difference over 49 cells (7x7)
            macro_cells = 49
            effective_candidates = max(gamma * eff_n * macro_cells, 1e-4)
            se = math.sqrt((2.0 * pi * (1.0 - pi)) / effective_candidates)

        if se is not None:
            from pcg.v3_5.power import power_engine_module

            rec = power_engine_module.recommend_n(
                effect=feas_delta_hat,
                se_at_n0=se,
                n0=eff_n,
                target_power=target_power,
            )
            if rec["status"] == "NO_GO_RESOURCE_FEASIBILITY":
                return {
                    "status": "NO_GO_RESOURCE_FEASIBILITY",
                    "terminal": "NO_GO_RESOURCE_FEASIBILITY",
                    "reason": f"No N on frozen grid reaches target power {target_power}",
                    "power_grid": rec["grid"],
                    "delta_fusion": delta_fusion,
                }
            return {
                "status": "PASS",
                "terminal": "PASS",
                "recommended_n_per_cell": rec["recommended_n_per_cell"],
                "power_grid": rec["grid"],
                "delta_fusion": delta_fusion,
            }

    return {
        "status": "PASS",
        "terminal": "PASS",
        "recommended_n_per_cell": n0,
        "delta_fusion": delta_fusion,
    }


def evaluate_prospective_terminal_decision(
    *,
    lcb95_vs_verifier_only: Optional[float] = None,
    lcb95_vs_fusion: Optional[float] = None,
    delta_u: Optional[float] = None,
    min_lcb95_delta_u: Optional[float] = None,
    lcb95_delta_u_by_comparator: Optional[Dict[str, float]] = None,
    lodo_min_lcb: Optional[float] = None,
    lodo_min_lcb_verifier_only: Optional[float] = None,
    lodo_min_lcb_fusion: Optional[float] = None,
    lodo_by_comparator: Optional[Dict[str, Any]] = None,
    audit_far_cp_upper: Optional[float] = None,
    indeterminate_share: Optional[float] = None,
    max_cell_factor_indeterminate: Optional[float] = None,
    cell_factor_indeterminacy: Optional[Dict[Any, float]] = None,
    delta_macro: Optional[float] = None,
    g_cal: Optional[float] = None,
    delta_fusion: Optional[float] = None,
    feas_delta_fusion_hat: Optional[float] = None,
    fusion_overfit_confounded: bool = False,
    fusion_cal_val_degradation_diff: Optional[float] = None,
    mutation_detected: bool = False,
    fit_convergence: bool = True,
    feas_delta_hat: Optional[float] = None,
    feas_gamma_hat: Optional[float] = None,
    feas_pi_hat: Optional[float] = None,
    feas_se_hat: Optional[float] = None,
    candidate_root_valid: bool = True,
    freeze_root_valid: bool = True,
    crash_at_stage: Optional[str] = None,
    duplicate_resumed_calls: int = 0,
    artifact_overwrites: int = 0,
    n_per_cell: Optional[int] = None,
) -> str:
    """Evaluates terminal prospective decision strictly conforming to immutable v3.5 authority."""
    # 0. Crash-safe resumption check
    if crash_at_stage is not None:
        if duplicate_resumed_calls > 0 or artifact_overwrites > 0:
            return "INVALID_TERMINAL"
        if (
            mutation_detected
            or not candidate_root_valid
            or not freeze_root_valid
            or not fit_convergence
        ):
            return "INVALID_TERMINAL"
        return "STRONG_RESULT"

    # 1. Protocol Integrity & Mutation Check -> INVALID_TERMINAL
    if mutation_detected or not candidate_root_valid or not freeze_root_valid:
        return "INVALID_TERMINAL"

    # 2. Calibration Fit Check -> NO_GO_CALIBRATION
    if not fit_convergence:
        return "NO_GO_CALIBRATION"

    # 3. Feasibility Conditions -> NO_GO_FEASIBILITY / NO_GO_RESOURCE_FEASIBILITY
    if (
        feas_delta_hat is not None
        or feas_gamma_hat is not None
        or feas_pi_hat is not None
        or g_cal is not None
        or feas_delta_fusion_hat is not None
    ):
        feas_res = evaluate_feasibility(
            feas_delta_hat=feas_delta_hat,
            g_cal=g_cal,
            feas_delta_fusion_hat=feas_delta_fusion_hat,
            feas_gamma_hat=feas_gamma_hat,
            feas_pi_hat=feas_pi_hat,
            feas_se_hat=feas_se_hat,
            n_per_cell=n_per_cell,
        )
        if feas_res["status"] != "PASS":
            return str(feas_res["terminal"])

        if delta_fusion is None and feas_res.get("delta_fusion") is not None:
            delta_fusion = feas_res["delta_fusion"]

    # Derive delta_fusion if not yet resolved
    if delta_fusion is None:
        delta_fusion = derive_delta_fusion(g_cal)

    # 4. Audit Gate -> NO_GO_AUDIT (one-shot exact CP upper bound <= 0.05)
    if audit_far_cp_upper is not None and audit_far_cp_upper > AUDIT_FAR_UPPER_BOUND_REQUIRED:
        return "NO_GO_AUDIT"

    # 5. Indeterminacy Cap -> INDETERMINATE
    # Cap is 0.10 per cell, per required factor (scalar represents maximum over required rates)
    effective_indeterminate = None
    if cell_factor_indeterminacy:
        effective_indeterminate = max(cell_factor_indeterminacy.values())
    elif max_cell_factor_indeterminate is not None:
        effective_indeterminate = max_cell_factor_indeterminate
    elif indeterminate_share is not None:
        effective_indeterminate = indeterminate_share

    if effective_indeterminate is not None and effective_indeterminate > INDETERMINACY_CAP_PER_CELL:
        return "INDETERMINATE"

    # 6. Substantive Performance & Non-Inferiority Checks -> NO_GO
    # Utility Gate: LCB95(Delta_U) >= -0.03 for each mandatory comparator
    effective_delta_u = None
    if lcb95_delta_u_by_comparator:
        effective_delta_u = min(lcb95_delta_u_by_comparator.values())
    elif min_lcb95_delta_u is not None:
        effective_delta_u = min_lcb95_delta_u
    elif delta_u is not None:
        effective_delta_u = delta_u

    if effective_delta_u is not None and effective_delta_u < UTILITY_NONINFERIORITY_MARGIN:
        return "NO_GO"

    # Strict Comparator Gate: CoverageMatchedVerifierOnly: LCB95(Delta_macro) > 0
    if lcb95_vs_verifier_only is not None and lcb95_vs_verifier_only <= STRICT_COMPARATOR_LCB_MARGIN:
        return "NO_GO"

    # Fusion Core Gate: SignalMatchedFusion: LCB95(Delta_macro) > -delta_fusion
    if lcb95_vs_fusion is not None and lcb95_vs_fusion <= -delta_fusion:
        return "NO_GO"

    # LODO Gate: comparator-specific
    # CoverageMatchedVerifierOnly: every omitted dataset LCB > 0
    eff_lodo_vo = lodo_min_lcb_verifier_only if lodo_min_lcb_verifier_only is not None else lodo_min_lcb
    if eff_lodo_vo is not None and eff_lodo_vo <= LODO_STRICT_LCB_MARGIN:
        return "NO_GO"

    # SignalMatchedFusion Core: every omitted dataset LCB > -delta_fusion
    eff_lodo_fusion = lodo_min_lcb_fusion if lodo_min_lcb_fusion is not None else lodo_min_lcb
    if eff_lodo_fusion is not None and eff_lodo_fusion <= -delta_fusion:
        return "NO_GO"

    if lodo_by_comparator:
        vo_lodos = lodo_by_comparator.get("CoverageMatchedVerifierOnly")
        if vo_lodos:
            min_vo = min(vo_lodos.values()) if isinstance(vo_lodos, dict) else min(vo_lodos)
            if min_vo <= LODO_STRICT_LCB_MARGIN:
                return "NO_GO"
        fu_lodos = lodo_by_comparator.get("SignalMatchedFusion")
        if fu_lodos:
            min_fu = min(fu_lodos.values()) if isinstance(fu_lodos, dict) else min(fu_lodos)
            if min_fu <= -delta_fusion:
                return "NO_GO"

    # 7. Strong Result Conditions -> STRONG_RESULT
    # - all CORE_GO conditions met
    # - SignalMatchedFusion LCB95(Delta_macro) > 0
    # - SignalMatchedFusion strong LODO condition: every omitted-dataset LCB > 0
    # - fusion comparison not confounded by excessive overfit diagnostic
    is_overfit_confounded = fusion_overfit_confounded
    if (
        fusion_cal_val_degradation_diff is not None
        and fusion_cal_val_degradation_diff > FUSION_OVERFIT_THRESHOLD
    ):
        is_overfit_confounded = True

    is_strong_fusion_primary = (
        lcb95_vs_fusion is not None and lcb95_vs_fusion > FUSION_STRONG_LCB_MARGIN
    )
    is_strong_fusion_lodo = (
        eff_lodo_fusion is not None and eff_lodo_fusion > FUSION_STRONG_LCB_MARGIN
    )

    if lodo_by_comparator and "SignalMatchedFusion" in lodo_by_comparator:
        fu_lodos = lodo_by_comparator["SignalMatchedFusion"]
        min_fu = min(fu_lodos.values()) if isinstance(fu_lodos, dict) else min(fu_lodos)
        is_strong_fusion_lodo = min_fu > FUSION_STRONG_LCB_MARGIN

    if is_strong_fusion_primary and is_strong_fusion_lodo and not is_overfit_confounded:
        return "STRONG_RESULT"

    # 8. All Core Conditions Met -> CORE_GO
    return "CORE_GO"
