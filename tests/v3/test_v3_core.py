"""v3.0 unit + property + contract tests. Offline; no model or network call."""
from __future__ import annotations
import json, math, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pcg.v3.canon import canonical_json, sha256_obj
from pcg.v3.channels import CHANNELS, CHANNEL_FIELD, CONJUNCTS, check, n_channels_fired
from pcg.v3.record import PerExampleRecord, json_schema
from pcg.v3.release import N_FIGURES, N_TABLES, WORKSTREAMS
from pcg.v3.stats.intervals import (clopper_pearson, clopper_pearson_lower,
                                    clopper_pearson_upper, betainc, hoeffding_halfwidth)
from pcg.v3.science.sv import sv_decomposition, assert_identity
from pcg.v3.science.audit import union_slack, stratified_envelope, StratumProbe, eps_tax_challenge
from pcg.v3.science.dependence import (EvidenceFloor, GateState, common_mode_floor,
                                       lambda_all_fail, rho_from_lambda, rho_ucb, u_joint)
from pcg.v3.science.responsibility import attribute, ranking_bound, tau_star_exact, tau_star_approx
from pcg.v3.science.controller import CostModel, regret_fixed_model, sensitivity, thresholds
from pcg.v3.science.shift import shift_alarm, transfer_bound
from pcg.v3.policy.local import LocalPolicyBackend
from pcg.v3.providers.registry import get_provider, NetworkCallBlocked
from pcg.v3.orchestration.graph import run_graph

# ------------------------------------------------------------------ canon
def test_canonical_json_key_order_invariant():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

def test_hash_stable_across_runs():
    assert sha256_obj({"x": [1, 2, {"z": None}]}) == sha256_obj({"x": [1, 2, {"z": None}]})

# ------------------------------------------------------------- certificate
def test_unknown_conjunct_is_failure():
    assert check(True, True, True, True) is True
    assert check(True, None, True, True) is False
    assert check(True, False, True, True) is False

def test_five_channels_and_four_conjuncts():
    assert len(CHANNELS) == 5 and len(CONJUNCTS) == 4

def test_n_channels_fired():
    assert n_channels_fired(int_fail=True, cov_gap=True) == 2

# ------------------------------------------------------------------ record
def test_record_validates_and_rejects():
    ok = PerExampleRecord(record_id="a", run_id="r", experiment_id="A03", system="pcg_mas")
    assert ok.validate() == []
    bad = PerExampleRecord(record_id="a", run_id="r", experiment_id="A03", system="pcg_mas",
                           check=True, v_h=True, v_pi=None, v_gamma=True, v_entail=True)
    assert bad.validate()

def test_refused_example_carries_no_pcg_loss():
    r = PerExampleRecord(record_id="a", run_id="r", experiment_id="A03", system="pcg_mas",
                         answered=False, loss_pcg=0.0)
    assert any("loss_pcg" in e for e in r.validate())

def test_schema_export_shape():
    s = json_schema()
    assert s["additionalProperties"] is False and len(s["properties"]) > 60

# --------------------------------------------------------------- intervals
def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(2, 10, 0.05)
    assert abs(lo - 0.0252) < 1e-3 and abs(hi - 0.5561) < 1e-3

def test_cp_lower_is_exactly_zero_at_zero_successes():
    assert clopper_pearson_lower(0, 500, 0.05) == 0.0

def test_betainc_matches_closed_form():
    assert abs(betainc(2, 3, 0.5) - 0.6875) < 1e-9

def test_hoeffding_monotone_in_n():
    assert hoeffding_halfwidth(1000, .05, 5) < hoeffding_halfwidth(100, .05, 5)

# ---------------------------------------------------------------- S/V exact
def test_sv_identity_exact_random():
    rng = random.Random(1)
    for _ in range(50):
        n = rng.randint(5, 200)
        lnc = [rng.random() for _ in range(n)]
        ans = [rng.random() < 0.8 for _ in range(n)]
        if not any(ans):
            ans[0] = True
        lpg = [(lnc[i] * rng.random() if ans[i] else None) for i in range(n)]
        r = sv_decomposition(lnc, lpg, ans)
        assert_identity(r)
        assert r.identity_residual <= 1e-12

def test_sv_full_coverage_zero_selectivity():
    lnc = [0.4, 0.6, 0.2]
    r = sv_decomposition(lnc, [0.1, 0.2, 0.05], [True] * 3)
    assert abs(r.S) < 1e-15

def test_sv_bound_on_S():
    rng = random.Random(2)
    lnc = [rng.random() for _ in range(100)]
    ans = [i % 3 != 0 for i in range(100)]
    lpg = [(lnc[i] * 0.5 if ans[i] else None) for i in range(100)]
    r = sv_decomposition(lnc, lpg, ans)
    assert abs(r.S) <= r.s_bound + 1e-12

# ------------------------------------------------------------- union slack
def test_union_slack_equals_expected_excess_multiplicity():
    rng = random.Random(3)
    recs = []
    for _ in range(500):
        fired = {c.value: False for c in CHANNELS}
        keys = list(fired)
        for k in rng.sample(keys, rng.randint(0, 3)):
            fired[k] = True
        recs.append({"int_fail": fired["IntFail"], "replay_fail": fired["ReplayFail"],
                     "drift_fail": fired["DriftFail"], "check_fail": fired["CheckFail"],
                     "cov_gap": fired["CovGap"]})
    u = union_slack(recs)
    assert abs((u["sum_marginals"] - u["pr_union"]) - u["lambda_union"]) < 1e-12

# ----------------------------------------------------------- audit envelope
def test_pi_unc_charged_once_not_per_channel():
    s = [StratumProbe("h1", 0.4, {CHANNEL_FIELD[c]: 100 for c in CHANNELS},
                      {CHANNEL_FIELD[c]: 1 for c in CHANNELS})]
    e = stratified_envelope(s, 0.05, eps_tax_cov=0.0)
    assert abs(e.pi_unc - 0.6) < 1e-12          # exactly once, not 5 x 0.6

def test_inner_sum_is_clipped_at_one():
    s = [StratumProbe("h1", 1.0, {CHANNEL_FIELD[c]: 10 for c in CHANNELS},
                      {CHANNEL_FIELD[c]: 9 for c in CHANNELS})]
    e = stratified_envelope(s, 0.05, eps_tax_cov=0.0)
    assert e.b_cov <= 1.0 + 1e-12 and e.clipped_strata == 1

def test_missing_probes_yield_undefined_not_zero():
    s = [StratumProbe("h1", 1.0, {}, {})]
    e = stratified_envelope(s, 0.05, eps_tax_cov=0.0)
    assert e.b_cov is None and e.contract_bad_bound is None   # undefined, never 0.0


def test_eps_tax_is_labelled_as_alarm():
    out = eps_tax_challenge([{"contract_bad": True}])
    assert out["eps_tax_chal"] == 1.0
    assert "not a deployment upper bound" in out["interpretation"]

# ------------------------------------------------------------- dependence
def test_lambda_undefined_when_marginal_zero():
    assert lambda_all_fail(0, [0, 3], 100) is None       # None, never 0.0

def test_rho_clamped_at_one():
    assert rho_from_lambda(0.001, 4) == 1.0

def test_gate_is_three_state_and_fails_closed_on_sparse_evidence():
    rng = random.Random(5)
    sparse = [[rng.random() < 0.001 for _ in range(4)] for _ in range(120)]
    r = rho_ucb(sparse, floor=EvidenceFloor(200, 5, 2), bar_rho=1.5)
    assert r.state is GateState.INSUFFICIENT_EVIDENCE
    assert r.state is not GateState.OPEN
    assert r.u_joint is not None                          # still reported for this exact k

def test_common_mode_detected():
    rng = random.Random(6)
    rows = [[True] * 4 if rng.random() < 0.06 else [rng.random() < 0.2 for _ in range(4)]
            for _ in range(600)]
    r = rho_ucb(rows, floor=EvidenceFloor(200, 5, 2), bar_rho=1.2, delta_tol=0.1)
    assert r.lambda_k is not None and r.lambda_k > 3.0
    assert r.state is GateState.CLOSED

def test_common_mode_floor_formula():
    f = common_mode_floor(0.05, 0.1)
    assert f["floor"] == 0.05 and f["k_star"] == math.ceil(math.log(0.05) / math.log(0.1))

def test_precision_requirement_fails_closed_when_inadequate():
    from pcg.v3.science.dependence import PrecisionRequirement, ImplementationEvidenceFloors
    rng = random.Random(7)
    rows = [[rng.random() < 0.10 for _ in range(3)] for _ in range(300)]
    # With a very tight max_ci_width requirement (e.g. 0.001), precision fails -> INSUFFICIENT_EVIDENCE
    strict_prec = PrecisionRequirement(max_ci_width=0.001, max_rel_width=0.001, enforce_precision=True)
    r = rho_ucb(rows, floor=ImplementationEvidenceFloors(100, 2, 2), precision=strict_prec, bar_rho=2.5)
    assert r.state is GateState.INSUFFICIENT_EVIDENCE
    assert r.precision_passed is False

def test_u_joint_defined_without_rho():
    assert 0.0 <= u_joint(0, 200, 0.05) <= 1.0

# --------------------------------------------------------- responsibility
def test_unresolved_when_margin_small():
    rng = random.Random(7)
    eff = {k: [max(-1, min(1, rng.gauss(0.4, 0.2))) for _ in range(25)] for k in "abc"}
    a = attribute(eff)
    assert a.unresolved is True and a.top1 is None

def test_tau_star_exact_beats_approximation():
    g, U, M = 0.4, 8, 200
    assert ranking_bound(g, U, M, tau_star_exact(g, U, M)) <= \
           ranking_bound(g, U, M, tau_star_approx(g, U, M)) + 1e-18

# -------------------------------------------------------------- controller
def test_all_four_actions_reachable():
    acts = {a for _, a in thresholds(CostModel())}
    assert acts == {"Answer", "Verify", "Escalate", "Refuse"}

def test_regret_respects_analytic_bound():
    rng = random.Random(8)
    rt = [rng.random() for _ in range(400)]
    rh = [max(0, min(1, r + rng.gauss(0, 0.05))) for r in rt]
    out = regret_fixed_model(CostModel(), rt, rh)
    assert out["bound_respected"] is True

def test_sensitivity_uses_per_theta_oracle():
    rng = random.Random(9)
    rh = [rng.random() for _ in range(200)]
    s = sensitivity({"nominal": CostModel(), "harm_heavy": CostModel(lam=3.0)}, "nominal", rh)
    assert s["R_max"] >= 0 and "per-theta oracle" in s["semantics"]

def test_a17_substream_theorem_separation():
    # A17-A has analytic_bound and bound_respected
    rng = random.Random(10)
    rt = [rng.random() for _ in range(100)]
    rh = [max(0, min(1, r + rng.gauss(0, 0.02))) for r in rt]
    a17_a = regret_fixed_model(CostModel(), rt, rh)
    assert "analytic_bound" in a17_a
    assert "bound_respected" in a17_a

    # A17-B is descriptive sensitivity only and must not provide analytic_bound
    a17_b = sensitivity({"nominal": CostModel(), "harm_heavy": CostModel(lam=3.0)}, "nominal", rh)
    assert "analytic_bound" not in a17_b
    assert "bound_respected" not in a17_b
    assert "per-theta oracle" in a17_b["semantics"]


# ------------------------------------------------------------------ shift
def test_alarm_is_lower_bound_and_never_a_bound_substitute():
    a = shift_alarm(0.85, 500, 500)
    assert 0.0 <= a.d_alarm <= 1.0 and a.d_bar is None
    assert transfer_bound(0.1, None) is None            # refuses to complete the bound

# ----------------------------------------------------------------- policy
def test_policy_fail_closed_and_isolation_clause():
    b = LocalPolicyBackend()
    assert b.evaluate({"actor": "p", "action": "a", "tool": "search"}).allowed is True
    assert b.evaluate({"actor": "p", "action": "a", "tool": "shell"}).allowed is False
    assert b.evaluate({"actor": "p", "action": "a", "verifier_context_shared": True}).allowed is False

# --------------------------------------------------------------- providers
def test_network_routes_blocked_offline():
    try:
        get_provider("hosted_provider", model_id="x")
        assert False, "network route should be blocked"
    except NetworkCallBlocked:
        pass

def test_offline_provider_is_deterministic():
    p = get_provider("offline_mock", seed=3)
    assert p.generate("abc").raw_output_sha256 == p.generate("abc").raw_output_sha256

# ------------------------------------------------------------ orchestration
def test_graph_runs_without_langgraph_and_is_deterministic():
    a = run_graph("q", run_id="r1")
    b = run_graph("q", run_id="r1")
    assert a.terminal == b.terminal == "accepted"
    assert len(a.provenance) == 11

def test_graph_refuses_on_failed_conjunct():
    st = run_graph("q", replay_check={"replay_ok": False})
    assert st.check is False and st.terminal == "refused"

# -------------------------------------------------------------- registry
def test_registry_matches_expected_counts():
    from pcg.v3.artifacts.registry import check_registry, load
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    c = check_registry(load(os.path.join(root, "manuscript_artifact_registry.json")))
    assert c["n_tables"] == N_TABLES and c["n_figures"] == N_FIGURES
    assert c["classes_valid"] and not c["tables_unmapped"] and not c["figures_unmapped"]

def test_workstream_catalog_complete():
    from pcg.v3.workstreams.catalog import CATALOG
    assert sorted(CATALOG) == list(WORKSTREAMS)
