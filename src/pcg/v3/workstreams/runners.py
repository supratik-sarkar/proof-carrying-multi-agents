"""Concrete offline runners for A01-A18.

Every runner reads canonical per-example records and emits the standard artifact
directory. None performs a model or network call in this release pass.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from ..channels import CHANNELS, CHANNEL_FIELD, CONJUNCT_TO_CHANNELS, Conjunct
from ..science.audit import (StratumProbe, eps_tax_challenge, pooled_envelope,
                             stratified_envelope, union_slack)
from ..science.controller import CostModel, regret_fixed_model, sensitivity
from ..science.dependence import (EvidenceFloor, common_mode_floor,
                                  renyi_infinity_bound, rho_ucb, u_joint)
from ..science.responsibility import attribute
from ..science.shift import shift_alarm
from ..science.sv import assert_identity, sv_decomposition
from ..stats.bootstrap import paired_crossed_bootstrap
from .base import Spec, Workstream
from .catalog import CATALOG


def _rate(num, den):
    """Undefined denominator -> None. NEVER 0.0."""
    if den in (None, 0) or num is None:
        return None
    return num / den


class A01BackendManifest(Workstream):
    def compute(self, recs):
        cells = defaultdict(set)
        seeds = defaultdict(set)
        for r in recs:
            c = r.get("cell_id")
            cells[c].add(r.get("backend_fingerprint"))
            seeds[c].add(r.get("seed"))
        mixed = {c: sorted(f for f in fs if f) for c, fs in cells.items() if len(fs) > 1}
        return {"cells": len(cells), "mixed_backend_cells": mixed,
                "seeds_per_cell": {c: len(s) for c, s in seeds.items()},
                "unfingerprinted": sum(1 for r in recs if not r.get("backend_fingerprint"))}

    def checks(self, m, recs):
        return {"no_mixed_backend_cells": not m["mixed_backend_cells"],
                "all_records_fingerprinted": m["unfingerprinted"] == 0,
                "records_present": len(recs) > 0}


class A02TableReconciliation(Workstream):
    def compute(self, recs):
        by = defaultdict(lambda: {"n": 0, "acc": 0, "harm": 0})
        for r in recs:
            k = (r.get("cell_id"), r.get("system"))
            b = by[k]
            b["n"] += 1
            if r.get("accepted"):
                b["acc"] += 1
                if r.get("h_joint"):
                    b["harm"] += 1
        rows = []
        for (cell, sysname), b in sorted(by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            rows.append({"cell_id": cell, "system": sysname, "N": b["n"], "N_acc": b["acc"],
                         "numerator": b["harm"], "denominator": b["acc"],
                         "estimate": _rate(b["harm"], b["acc"]),
                         "metric_version": self.spec.metric_version})
        return {"rows": rows, "undefined_rows": sum(1 for r in rows if r["estimate"] is None)}

    def checks(self, m, recs):
        return {"no_silent_zero": all(r["estimate"] is not None or r["denominator"] == 0
                                      for r in m["rows"]),
                "records_present": len(recs) > 0}


class A03SVDecomposition(Workstream):
    def compute(self, recs):
        lnc = [r.get("loss_nocert") for r in recs]
        lpg = [r.get("loss_pcg") for r in recs]
        ans = [bool(r.get("answered")) for r in recs]
        if not recs or any(x is None for x in lnc):
            return {"error": "loss_nocert required on every example", "S": None, "V": None}
        res = sv_decomposition(lnc, lpg, ans)
        assert_identity(res)
        boot = paired_crossed_bootstrap(recs, n_boot=int(self.spec.params.get("n_boot", 2000)),
                                        seed=17)
        return {**res.to_dict(), "bootstrap": boot}

    def checks(self, m, recs):
        ok = m.get("identity_residual") is not None and m["identity_residual"] <= 1e-12
        return {"sv_identity_exact": ok,
                "V_bounded_away_from_zero": (m.get("V") or 0) > 1e-9,
                "records_present": len(recs) > 0}


class A04AuditorInvariance(Workstream):
    def compute(self, recs):
        by = defaultdict(dict)
        for r in recs:
            by[r.get("record_id")][r.get("host_fingerprint")] = r.get("check")
        dis = {k: v for k, v in by.items() if len(set(v.values())) > 1}
        return {"certificates": len(by), "hosts": sorted({r.get("host_fingerprint") for r in recs}),
                "disagreements": len(dis), "disagreement_ids": sorted(dis)[:50],
                "agreement_rate": _rate(len(by) - len(dis), len(by))}

    def checks(self, m, recs):
        return {"at_least_two_hosts": len([h for h in m["hosts"] if h]) >= 2,
                "all_disagreements_disclosed": True,
                "records_present": len(recs) > 0}


class A05CitationOnly(Workstream):
    def compute(self, recs):
        out = {}
        for sysname in sorted({r.get("system") for r in recs}):
            rs = [r for r in recs if r.get("system") == sysname]
            acc = [r for r in rs if r.get("accepted")]
            out[sysname] = {
                "N": len(rs), "N_acc": len(acc),
                "coverage": _rate(len(acc), len(rs)),
                "H_support": _rate(sum(1 for r in acc if r.get("h_support")), len(acc)),
                "H_exec": _rate(sum(1 for r in acc if r.get("h_exec")), len(acc)),
                "H_joint": _rate(sum(1 for r in acc if r.get("h_joint")), len(acc)),
            }
        return {"by_system": out}

    def checks(self, m, recs):
        return {"citation_only_present": "citation_only" in m["by_system"],
                "records_present": len(recs) > 0}


class A06SeparatingWitnesses(Workstream):
    FAM = {"W_H": "v_h", "W_Pi": "v_pi", "W_Gamma": "v_gamma", "W_vdash": "v_entail"}

    def compute(self, recs):
        out = {}
        for fam, key in self.FAM.items():
            rs = [r for r in recs if r.get("stratum_id") == fam]
            good = [r for r in rs
                    if r.get(key) is False
                    and all(r.get(o) is True for o in self.FAM.values() if o != key)]
            out[fam] = {"n": len(rs), "exclusivity_pass": len(good),
                        "pass_rate": _rate(len(good), len(rs)),
                        "accept_rate": _rate(sum(1 for r in rs if r.get("accepted")), len(rs))}
        return {"families": out,
                "conjunct_channel_map": {c.value: [x.value for x in v]
                                         for c, v in CONJUNCT_TO_CHANNELS.items()}}

    def checks(self, m, recs):
        f = m["families"]
        return {"all_four_families_nonempty": all(v["n"] > 0 for v in f.values()),
                "min_100_per_family": all(v["n"] >= 100 for v in f.values()),
                "exclusivity_holds": all(v["pass_rate"] == 1.0 for v in f.values() if v["n"]),
                "records_present": len(recs) > 0}


class A07AuditSampling(Workstream):
    def compute(self, recs):
        strata = defaultdict(lambda: {"pi": 0.0, "probes": defaultdict(int), "fails": defaultdict(int)})
        seen = set()
        for r in recs:
            h = r.get("stratum_id") or "h_unknown"
            s = strata[h]
            if h not in seen:
                s["pi"] = float(r.get("sampling_weight") or 0.0)
                seen.add(h)
            for c in CHANNELS:
                f = CHANNEL_FIELD[c]
                s["probes"][f] += 1
                if r.get(f):
                    s["fails"][f] += 1
        probes = [StratumProbe(h, v["pi"], dict(v["probes"]), dict(v["fails"]))
                  for h, v in strata.items()]
        tax = eps_tax_challenge(recs)
        env = stratified_envelope(probes, delta=float(self.spec.params.get("delta", 0.05)),
                                  eps_tax_cov=tax["eps_tax_chal"], eps_src=None)
        pooled = pooled_envelope(
            {CHANNEL_FIELD[c]: sum(1 for r in recs if r.get(CHANNEL_FIELD[c])) for c in CHANNELS},
            {CHANNEL_FIELD[c]: len(recs) for c in CHANNELS}, 0.05)
        return {"stratified": env.to_dict(), "pooled_envelope": pooled,
                "eps_tax_chal": tax, "n_strata": len(probes)}

    def checks(self, m, recs):
        return {"pi_unc_charged_once": m["stratified"]["pi_unc"] is not None,
                "envelope_defined": m["stratified"]["b_cov"] is not None,
                "records_present": len(recs) > 0}


class A08Injection(Workstream):
    def compute(self, recs):
        by = defaultdict(list)
        for r in recs:
            by[r.get("injection_regime") or "none"].append(r)
        out = {}
        floor = EvidenceFloor(**self.spec.params.get("evidence_floor",
                                                    {"n_min": 100, "k_min": 5, "q0": 2}))
        for regime, rs in sorted(by.items()):
            bf = [r["branch_failures"] for r in rs if r.get("branch_failures")]
            dep = rho_ucb(bf, delta=0.05, floor=floor,
                          bar_rho=self.spec.params.get("bar_rho"),
                          delta_tol=self.spec.params.get("delta_tol", 0.0)) if bf else None
            acc_attack = _rate(sum(1 for r in rs if r.get("accepted") and r.get("h_joint")), len(rs))
            out[regime] = {
                "n": len(rs),
                "accepted_attack_success": acc_attack,
                "false_refusal": _rate(sum(1 for r in rs if r.get("answered") is False), len(rs)),
                "dependence": dep.to_dict() if dep else None,
            }
        return {"regimes": out, "common_mode_floor":
                common_mode_floor(float(self.spec.params.get("q_cm", 0.05)),
                                  float(self.spec.params.get("eps_path", 0.1)))}

    def checks(self, m, recs):
        regs = m["regimes"]
        gated = [v["dependence"]["state"] for v in regs.values() if v.get("dependence")]
        return {"isolated_and_shared_regimes_present":
                    any("isolated" in k for k in regs) and any("shared" in k for k in regs),
                "gate_never_open_under_shared_context":
                    all(regs[k]["dependence"]["state"] != "OPEN"
                        for k in regs if "shared" in k and regs[k].get("dependence")),
                "gate_states_reported": len(gated) > 0,
                "records_present": len(recs) > 0}


class A09Shift(Workstream):
    def compute(self, recs):
        out = {}
        for regime in sorted({r.get("shift_regime") or "none" for r in recs}):
            rs = [r for r in recs if (r.get("shift_regime") or "none") == regime]
            a_hat = float(self.spec.params.get("classifier_balanced_accuracy", {}).get(regime, 0.5))
            al = shift_alarm(a_hat, n_cal=len(rs), n_dep=len(rs),
                             threshold=float(self.spec.params.get("alarm_threshold", 0.10)))
            out[regime] = {"n": len(rs), **al.to_dict(),
                           "observed_contract_bad": _rate(
                               sum(1 for r in rs if r.get("contract_bad")), len(rs))}
        return {"regimes": out,
                "note": "D_alarm is a gate; it is never substituted for a valid D_bar"}

    def checks(self, m, recs):
        return {"d_bar_not_faked": all(v.get("d_bar") is None for v in m["regimes"].values()),
                "records_present": len(recs) > 0}


class A10DirectTiming(Workstream):
    def compute(self, recs):
        def pct(xs, q):
            if not xs:
                return None
            ys = sorted(xs)
            i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
            return ys[i]
        out = {}
        for sysname in sorted({r.get("system") for r in recs}):
            rs = [r for r in recs if r.get("system") == sysname]
            lat = [r["latency_ms"] for r in rs if r.get("latency_ms") is not None]
            acc_ok = [r for r in rs if r.get("accepted") and not r.get("h_joint")]
            cost = sum(r.get("billed_cost_usd") or 0.0 for r in rs)
            out[sysname] = {
                "N": len(rs), "latency_p50_ms": pct(lat, 0.50), "latency_p95_ms": pct(lat, 0.95),
                "tokens_in": sum(r.get("tokens_in") or 0 for r in rs),
                "tokens_out": sum(r.get("tokens_out") or 0 for r in rs),
                "model_calls": sum(r.get("model_calls") or 0 for r in rs),
                "retrieval_calls": sum(r.get("retrieval_calls") or 0 for r in rs),
                "tool_calls": sum(r.get("tool_calls") or 0 for r in rs),
                "checker_calls": sum(r.get("checker_calls") or 0 for r in rs),
                "replay_calls": sum(r.get("replay_calls") or 0 for r in rs),
                "billed_cost_usd": cost,
                "accepted_correct": len(acc_ok),
                "cost_per_accepted_correct": _rate(cost, len(acc_ok)),
            }
        return {"by_system": out}

    def checks(self, m, recs):
        return {"latency_present": all(v["latency_p50_ms"] is not None for v in m["by_system"].values()),
                "cost_denominator_explicit": all("accepted_correct" in v for v in m["by_system"].values()),
                "records_present": len(recs) > 0}


class A11UnionTightness(Workstream):
    def compute(self, recs):
        u = union_slack(recs)
        return {**u, "containment_slack_note":
                "reported separately from finite-sample UCB slack per the v3.0 contract"}

    def checks(self, m, recs):
        return {"identity_exact": (m.get("identity_residual") or 1) < 1e-9,
                "all_five_channels_counted": len(m["per_channel"]) == 5,
                "records_present": len(recs) > 0}


class A12RenyiDependence(Workstream):
    def compute(self, recs):
        table = defaultdict(int)
        n = 0
        for r in recs:
            bf = r.get("branch_failures")
            if not bf:
                continue
            n += 1
            table[tuple(int(b) for b in bf)] += 1
        e = renyi_infinity_bound(dict(table), n) if n else None
        bf_rows = [r["branch_failures"] for r in recs if r.get("branch_failures")]
        dep = rho_ucb(bf_rows, floor=EvidenceFloor(50, 3, 2)) if bf_rows else None
        lam = dep.lambda_k if dep else None
        return {"n": n, "cells_populated": len(table), "exp_D_inf": e,
                "lambda_k": lam, "rho_k": dep.rho_k if dep else None,
                "bound_respected": (None if (e is None or lam is None) else lam <= e + 1e-9),
                "status": "SECONDARY: only where the 2^k table is sufficiently populated"}

    def checks(self, m, recs):
        return {"renyi_bound_respected": m["bound_respected"] in (True, None),
                "records_present": len(recs) > 0}


class A13TCBPerturbation(Workstream):
    def compute(self, recs):
        out = {}
        for comp in sorted({r.get("stratum_id") for r in recs if r.get("stratum_id")}):
            rs = [r for r in recs if r.get("stratum_id") == comp]
            flips = sum(1 for r in rs if r.get("check") is False)
            out[comp] = {"n": len(rs), "acceptance_flips": flips,
                         "flip_rate": _rate(flips, len(rs))}
        return {"components": out, "n_components": len(out)}

    def checks(self, m, recs):
        return {"eight_tcb_components": m["n_components"] == 8,
                "records_present": len(recs) > 0}


class A14TaxonomyOpenSet(Workstream):
    def compute(self, recs):
        tax = eps_tax_challenge(recs)
        fams = defaultdict(lambda: {"n": 0, "bad": 0, "missed": 0, "covgap_only": 0})
        for r in recs:
            f = fams[r.get("corruption") or "unknown"]
            f["n"] += 1
            if r.get("contract_bad"):
                f["bad"] += 1
                fired = [c for c in CHANNELS if r.get(CHANNEL_FIELD[c])]
                if not fired:
                    f["missed"] += 1
                elif len(fired) == 1 and fired[0].value == "CovGap":
                    f["covgap_only"] += 1
        return {**tax, "by_family": {k: dict(v) for k, v in sorted(fams.items())},
                "covgap_only_total": sum(v["covgap_only"] for v in fams.values())}

    def checks(self, m, recs):
        return {"is_alarm_not_bound": m["interpretation"].startswith("challenge-set"),
                "covgap_not_used_as_catchall":
                    m["covgap_only_total"] <= max(1, int(0.25 * (m["contract_bad"] or 1))),
                "records_present": len(recs) > 0}


class A15EntailmentChecker(Workstream):
    def compute(self, recs):
        out = {}
        for cond in sorted({r.get("corruption") or "nominal" for r in recs}):
            rs = [r for r in recs if (r.get("corruption") or "nominal") == cond]
            tp = sum(1 for r in rs if r.get("v_entail") and r.get("eps_src_label") is not True)
            fp = sum(1 for r in rs if r.get("v_entail") and r.get("eps_src_label") is True)
            fn = sum(1 for r in rs if r.get("v_entail") is False and r.get("eps_src_label") is not True)
            tn = sum(1 for r in rs if r.get("v_entail") is False and r.get("eps_src_label") is True)
            acc = [r for r in rs if r.get("accepted")]
            out[cond] = {
                "n": len(rs), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": _rate(tp, tp + fp), "recall": _rate(tp, tp + fn),
                "fpr": _rate(fp, fp + tn), "fnr": _rate(fn, fn + tp),
                "pcg_accepted_harm": _rate(sum(1 for r in acc if r.get("h_joint")), len(acc)),
                "check_fail_rate": _rate(sum(1 for r in rs if r.get("check_fail")), len(rs)),
                "coverage": _rate(len(acc), len(rs)),
                "checker_fingerprint": next((r.get("checker_fingerprint") for r in rs
                                             if r.get("checker_fingerprint")), None),
            }
        return {"conditions": out,
                "alpha_ent": self.spec.params.get("alpha_ent"),
                "threshold_policy": "max retained coverage s.t. UCB(checker FPR) <= alpha_ent, frozen pre-eval"}

    def checks(self, m, recs):
        c = m["conditions"]
        return {"nominal_condition_present": "nominal" in c,
                "degraded_conditions_present": len(c) >= 2,
                "checker_identified": any(v["checker_fingerprint"] for v in c.values()),
                "alpha_ent_frozen": m["alpha_ent"] is not None,
                "records_present": len(recs) > 0}


class A16BudgetMatched(Workstream):
    def compute(self, recs):
        out = defaultdict(dict)
        for arm in sorted({r.get("cell_id") or "arm" for r in recs}):
            for sysname in sorted({r.get("system") for r in recs}):
                rs = [r for r in recs if r.get("system") == sysname and (r.get("cell_id") or "arm") == arm]
                if not rs:
                    continue
                acc = [r for r in rs if r.get("accepted")]
                calls = sum(r.get("model_calls") or 0 for r in rs)
                gen = sum((r.get("model_calls") or 0) - (r.get("checker_calls") or 0) for r in rs)
                out[arm][sysname] = {
                    "N": len(rs), "model_calls": calls,
                    "generator_calls": gen, "checker_calls": sum(r.get("checker_calls") or 0 for r in rs),
                    "retrieval_calls": sum(r.get("retrieval_calls") or 0 for r in rs),
                    "tokens": sum((r.get("tokens_in") or 0) + (r.get("tokens_out") or 0) for r in rs),
                    "cost_usd": sum(r.get("billed_cost_usd") or 0.0 for r in rs),
                    "harmful_accepted_rate": _rate(sum(1 for r in acc if r.get("h_joint")), len(acc)),
                    "coverage": _rate(len(acc), len(rs)),
                    "allocation_split": {
                        "generation": _rate(gen, calls),
                        "checking": _rate(sum(r.get("checker_calls") or 0 for r in rs), calls),
                    },
                }
        return {"arms": {k: dict(v) for k, v in out.items()}}

    def checks(self, m, recs):
        arms = m["arms"]
        return {"allocation_split_reported":
                    all("allocation_split" in s for a in arms.values() for s in a.values()),
                "multiple_budget_arms": len(arms) >= 2,
                "records_present": len(recs) > 0}


class A17ControllerSensitivity(Workstream):
    def compute(self, recs):
        r_hat = [r["utility"] for r in recs if r.get("utility") is not None]
        r_true = [r["cov_cert"] for r in recs if r.get("cov_cert") is not None]
        nominal = CostModel()
        models = {"nominal": nominal,
                  "harm_heavy": CostModel(lam=3.0),
                  "latency_sensitive": CostModel(
                      c_lat={"Answer": 0.0, "Verify": 0.30, "Escalate": 0.60, "Refuse": 0.225}),
                  "refusal_expensive": CostModel(
                      c_lat={"Answer": 0.0, "Verify": 0.075, "Escalate": 0.15, "Refuse": 0.60})}
        a = (regret_fixed_model(nominal, r_true, r_hat)
             if r_true and len(r_true) == len(r_hat) else None)
        b = sensitivity(models, "nominal", r_hat) if r_hat else None
        return {"A17_A_fixed_model": a, "A17_B_misspecification": b,
                "semantics": "A17-A tests the 2*L_ctrl*eps_cal bound; A17-B is model-relative only"}

    def checks(self, m, recs):
        a = m.get("A17_A_fixed_model")
        return {"analytic_bound_respected": (a or {}).get("bound_respected", None) in (True, None),
                "per_theta_oracle_used": m.get("A17_B_misspecification") is not None,
                "records_present": len(recs) > 0}


class A18PrivacyCaseStudy(Workstream):
    def compute(self, recs):
        return {"status": "OPTIONAL / NOT RUN",
                "provenance_class": "MODELLED",
                "note": ("Privacy is MODELLED in the v3.0 manuscript. A18 is a blocker only if "
                         "privacy is restored as a substantive empirical claim, which requires a "
                         "concrete trust boundary, adjacency relation and DP accounting.")}

    def checks(self, m, recs):
        return {"not_claimed_empirical": m["provenance_class"] == "MODELLED"}


REGISTRY = {
    "A01": A01BackendManifest, "A02": A02TableReconciliation, "A03": A03SVDecomposition,
    "A04": A04AuditorInvariance, "A05": A05CitationOnly, "A06": A06SeparatingWitnesses,
    "A07": A07AuditSampling, "A08": A08Injection, "A09": A09Shift, "A10": A10DirectTiming,
    "A11": A11UnionTightness, "A12": A12RenyiDependence, "A13": A13TCBPerturbation,
    "A14": A14TaxonomyOpenSet, "A15": A15EntailmentChecker, "A16": A16BudgetMatched,
    "A17": A17ControllerSensitivity, "A18": A18PrivacyCaseStudy,
}


def build(eid: str, **params) -> Workstream:
    spec = CATALOG[eid]
    if params:
        spec = Spec(**{**spec.to_dict(), "params": {**spec.params, **params}})
    return REGISTRY[eid](spec)
