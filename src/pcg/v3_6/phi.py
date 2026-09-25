"""Canonical non-circular phi(R) constructor. One implementation, FIT and DEV.

The phi contract is documented in pcg.v3_5.acceptance.reconstruct_pcg_acceptance_from_phi:
  min_critical_S  min S_i over critical obligations
  max_critical_K  max K_i over critical obligations
  supp_fraction   fraction of supplementary obligations certified
  v_h_pass / v_gamma_pass / v_pi_pass   factor states as 1.0/0.0

The primitives come from pcg.v3_5.obligations.evaluate_obligations_with_trace,
which already emits {obligation_id: {S_i, K_i, is_critical}}. Nothing here reads
the acceptance decision: a phi derived from pcg_accepted would make
SignalMatchedFusion a proxy for PCG and the comparison circular.
"""
from typing import Any, Dict, Mapping, Optional

PHI_VERSION = "PCG_MAS_V3_6_PHI_V1"
FEATURE_NAMES = ("min_critical_S", "max_critical_K", "supp_fraction",
                 "v_h_pass", "v_gamma_pass", "v_pi_pass")
FORBIDDEN_SOURCES = ("pcg_accepted", "accepted", "harm", "harm_label", "gold",
                     "reference", "evaluator_label", "check_result", "passed")


class PhiConstructionError(RuntimeError):
    pass


def _state_to_float(state: Any, *, treat_na_as_pass: bool) -> float:
    s = getattr(state, "value", state)
    s = str(s).upper()
    if s == "PASS":
        return 1.0
    if s == "NOT_APPLICABLE":
        return 1.0 if treat_na_as_pass else 0.0
    return 0.0


def phi_from_obligation_scores(obligation_scores: Mapping[str, Mapping[str, float]],
                               *, tau_E: float, tau_C: float,
                               v_h, v_gamma, v_pi,
                               is_replay_applicable: bool) -> Dict[str, float]:
    """Build phi from obligation-level S/K and factor states only."""
    if not isinstance(obligation_scores, Mapping):
        raise PhiConstructionError("OBLIGATION_SCORES_NOT_MAPPING")
    crit, supp = [], []
    for oid, sc in obligation_scores.items():
        if "S_i" not in sc or "K_i" not in sc:
            raise PhiConstructionError(f"OBLIGATION_MISSING_S_OR_K:{oid}")
        (crit if bool(sc.get("is_critical", 1.0)) else supp).append(sc)

    # An empty critical set is vacuously satisfied by the gate in
    # acceptance.evaluate_semantic_gate, so it maps to the satisfying extreme
    # rather than to the fail-closed defaults used for *missing* keys.
    min_critical_S = min((float(s["S_i"]) for s in crit), default=1.0)
    max_critical_K = max((float(s["K_i"]) for s in crit), default=0.0)
    if supp:
        certified = sum(1 for s in supp
                        if float(s["S_i"]) >= tau_E and float(s["K_i"]) <= tau_C)
        supp_fraction = certified / len(supp)
    else:
        supp_fraction = 1.0

    return {
        "min_critical_S": float(min_critical_S),
        "max_critical_K": float(max_critical_K),
        "supp_fraction": float(supp_fraction),
        "v_h_pass": _state_to_float(v_h, treat_na_as_pass=False),
        "v_gamma_pass": _state_to_float(v_gamma, treat_na_as_pass=False),
        # V_Pi is NOT_APPLICABLE on non-replay datasets and the gate ignores it
        # there, so NA maps to satisfying exactly when replay is inapplicable.
        "v_pi_pass": _state_to_float(v_pi, treat_na_as_pass=not is_replay_applicable),
    }


def phi_from_certificate(certification_record: Mapping[str, Any], *,
                         tau_E: float, tau_C: float,
                         is_replay_applicable: Optional[bool] = None) -> Dict[str, float]:
    """Canonical entry point. Used byte-identically for FIT and DEV."""
    assert_no_forbidden_source(certification_record)
    scores = certification_record.get("obligation_scores")
    if not scores:
        raise PhiConstructionError(
            "OBLIGATION_SCORES_ABSENT: certification record must persist obligation-level "
            "S_i/K_i; phi may not be reconstructed from the acceptance decision")
    factors = certification_record.get("factors") or {}
    if is_replay_applicable is None:
        is_replay_applicable = str(certification_record.get("dataset_id")) in ("toolbench", "weblinx")
    return phi_from_obligation_scores(
        scores, tau_E=tau_E, tau_C=tau_C,
        v_h=factors.get("V_H"), v_gamma=factors.get("V_Gamma"), v_pi=factors.get("V_Pi"),
        is_replay_applicable=bool(is_replay_applicable))


def assert_no_forbidden_source(record: Mapping[str, Any]) -> None:
    """phi must never be derived from acceptance or evaluator labels."""
    scores = record.get("obligation_scores")
    if scores is None:
        return
    for oid, sc in scores.items():
        for k in sc:
            if str(k).lower() in FORBIDDEN_SOURCES:
                raise PhiConstructionError(f"CIRCULAR_PHI_SOURCE:{oid}.{k}")


def assert_phi_wellformed(phi: Mapping[str, float]) -> bool:
    missing = [f for f in FEATURE_NAMES if f not in phi]
    if missing:
        raise PhiConstructionError(f"PHI_MISSING_FEATURES:{missing}")
    for f in FEATURE_NAMES:
        v = phi[f]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise PhiConstructionError(f"PHI_NON_NUMERIC:{f}={v!r}")
        if not (0.0 <= float(v) <= 1.0):
            raise PhiConstructionError(f"PHI_OUT_OF_RANGE:{f}={v}")
    return True
