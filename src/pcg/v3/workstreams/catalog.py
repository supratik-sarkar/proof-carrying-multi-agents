"""A01-A18 catalogue: frozen identity, tier, provenance class, model-call need."""
from __future__ import annotations
from typing import Dict
from .base import Spec

# tier: T0 = paper does not exist without it; T1 = theory needs it; T2 = strengthens
# NOTE: every value in `params` is FROZEN PRE-REGISTRATION. Changing one after
# observing results is a spec violation and `verify_spec` will refuse the run.
CATALOG: Dict[str, Spec] = {
    "A01": Spec("A01", "backend_manifest",         "PROTOCOL", True,  "T0"),
    "A02": Spec("A02", "table_reconciliation",     "DERIVED",  False, "T0"),
    "A03": Spec("A03", "sv_decomposition",         "DERIVED",  False, "T0"),
    "A04": Spec("A04", "auditor_invariance",       "DIRECT",   False, "T1"),
    "A05": Spec("A05", "citation_only",            "DIRECT",   True,  "T1"),
    "A06": Spec("A06", "separating_witnesses",     "DIRECT",   False, "T2"),
    "A07": Spec("A07", "audit_sampling",           "DERIVED",  False, "T1",
                params={"delta": 0.05, "n_min": 200}),
    "A08": Spec("A08", "injection",                "DIRECT",   True,  "T2",
                params={"bar_rho": 1.35, "delta_tol": 0.15, "q_cm": 0.05, "eps_path": 0.18,
                        "evidence_floor": {"n_min": 200, "k_min": 5, "q0": 2}}),
    "A09": Spec("A09", "shift",                    "DIRECT",   True,  "T2",
                params={"alarm_threshold": 0.10,
                        "classifier_balanced_accuracy": {"none": 0.51, "held_out_dataset": 0.72,
                                                          "backend_change": 0.66, "corruption": 0.81,
                                                          "tool_drift": 0.58}}),
    "A10": Spec("A10", "direct_r5_timing",         "DIRECT",   True,  "T1"),
    "A11": Spec("A11", "union_tightness",          "DERIVED",  False, "T2"),
    "A12": Spec("A12", "renyi_dependence",         "DERIVED",  False, "T2"),
    "A13": Spec("A13", "tcb_perturbation",         "DIRECT",   False, "T2"),
    "A14": Spec("A14", "taxonomy_open_set",        "DIRECT",   False, "T1"),
    "A15": Spec("A15", "entailment_checker",       "DIRECT",   True,  "T0",
                params={"alpha_ent": 0.05, "threshold_rule": "max_coverage_s.t._UCB(FPR)<=alpha_ent"}),
    "A16": Spec("A16", "budget_matched_ensembles", "DIRECT",   True,  "T0"),
    "A17": Spec("A17", "controller_sensitivity",   "DERIVED",  False, "T2"),
    "A18": Spec("A18", "privacy_case_study",       "MODELLED", False, "OPTIONAL"),
}
IDS = tuple(sorted(CATALOG))
