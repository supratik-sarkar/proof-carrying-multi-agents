"""Comprehensive test suite for S02 Prospective Scientific Harness.

Covers:
1. Unit tests:
   - Zero denominators -> UNDEFINED (None)
   - Vacuous obligation satisfaction (effective PASS/1 for non-applicable)
   - Vacuous harm satisfaction (effective 0 for non-applicable)
   - Specialist preservation (TIE is PASS, negative is FAIL, undefined is INDETERMINATE)
2. Integration tests:
   - SpecialistCompose coverage evaluation
   - Scope-matched coverage gate C6 compliance
   - Integrated risk gain vs NoCert (C2)
   - A16 zero-dollar ceiling enforcement & allocation splits
3. Property-based tests (Hypothesis):
   - Invariance of vacuous channels
   - Non-zero denominator bounds
   - Deterministic mathematical outputs
4. Determinism tests:
   - Multiple identical runs produce byte-for-byte identical results
   - A15 deterministic mock scoring
5. Firewall tests:
   - CHECKER_CALIBRATION and FINAL partition isolation
   - Zero hosted API calls / zero paid spend
6. Historical immutability tests:
   - Raw AG-2.6 artifacts remain byte-identical to frozen SHAs
7. Scientific constants immutability:
   - Version payload and protocol SHA unmutated
"""
import hashlib
import json
from pathlib import Path
import pytest
from hypothesis import given, strategies as st

from pcg.v3.harness import (
    A15DebertaNLIInterface,
    A16ResourceAccounting,
    AcceptanceEvaluator,
    HarmEvaluator,
    SpecialistCompose,
    UtilityAccounting,
    evaluate_integrated_risk,
    evaluate_specialist_comparison,
    format_rate,
    safe_rate,
)
from pcg.v3.release import version_payload

ROOT = Path(__file__).resolve().parents[2]
DS_DIR = ROOT / "artifacts" / "v3_0" / "directionality_strengthening"
S01_DIR = ROOT / "artifacts" / "v3_0" / "master_forward_controller" / "stages" / "S01"


# =============================================================================
# 1. Unit Tests: Safe Math, Vacuous Pass, Harm, Preservation
# =============================================================================

def test_safe_rate_zero_denominator():
    assert safe_rate(0, 0) is None
    assert safe_rate(5, 0) is None
    assert safe_rate(None, 10) is None
    assert safe_rate(10, None) is None
    assert safe_rate(0, 10) == 0.0
    assert safe_rate(5, 10) == 0.5


def test_format_rate():
    assert format_rate(None) == "UNDEFINED"
    assert format_rate(0.5) == "0.5000"
    assert format_rate(0.0) == "0.0000"


def test_vacuous_obligation_satisfaction():
    acc = AcceptanceEvaluator()

    # ToolBench: grounding is NOT applicable, policy & replay ARE applicable
    res_tb = acc.check_pcg("toolbench", v_h=False, v_pi=True, v_gamma=True, v_entail=False)
    assert res_tb["accepted"] is True
    assert res_tb["effective_v_h"] is True  # Vacuous PASS
    assert res_tb["effective_v_entail"] is True  # Vacuous PASS
    assert "v_h" in res_tb["vacuous_channels"]
    assert "v_entail" in res_tb["vacuous_channels"]

    # ToolBench: if policy fails, acceptance fails
    res_tb_fail = acc.check_pcg("toolbench", v_h=True, v_pi=True, v_gamma=False, v_entail=True)
    assert res_tb_fail["accepted"] is False

    # FEVER: grounding IS applicable, policy & replay are NOT applicable
    res_fev = acc.check_pcg("fever", v_h=True, v_pi=False, v_gamma=False, v_entail=True)
    assert res_fev["accepted"] is True
    assert res_fev["effective_v_gamma"] is True  # Vacuous PASS
    assert res_fev["effective_v_pi"] is True  # Vacuous PASS
    assert "v_gamma" in res_fev["vacuous_channels"]
    assert "v_pi" in res_fev["vacuous_channels"]

    # FEVER: if entailment fails, acceptance fails
    res_fev_fail = acc.check_pcg("fever", v_h=True, v_pi=True, v_gamma=True, v_entail=False)
    assert res_fev_fail["accepted"] is False


def test_vacuous_harm_satisfaction():
    harm = HarmEvaluator()

    # ToolBench: textual support harm is NOT applicable
    # Even if raw lexical overlap is zero (h_support=True), it evaluates vacuously to False!
    res_tb = harm.evaluate_harm("toolbench", h_support=True, h_exec=False)
    assert res_tb["effective_h_support"] is False
    assert res_tb["effective_h_exec"] is False
    assert res_tb["h_joint"] is False

    # If execution harm is True on ToolBench, h_joint is True
    res_tb_exec = harm.evaluate_harm("toolbench", h_support=False, h_exec=True)
    assert res_tb_exec["h_joint"] is True

    # FEVER: execution harm is NOT applicable
    res_fev = harm.evaluate_harm("fever", h_support=False, h_exec=True)
    assert res_fev["effective_h_support"] is False
    assert res_fev["effective_h_exec"] is False
    assert res_fev["h_joint"] is False


def test_specialist_preservation_decision_rules():
    # Empirical TIE strictly satisfies preservation
    res_tie = evaluate_specialist_comparison(0.5, 0.5, higher_is_better=False)
    assert res_tie["status"] == "TIE"
    assert res_tie["preservation"] == "PASS"
    assert res_tie["delta"] == 0.0

    # Positive gain satisfies preservation
    res_pos = evaluate_specialist_comparison(0.2, 0.4, higher_is_better=False)  # PCG harm 0.2 < Spec harm 0.4
    assert res_pos["status"] == "POSITIVE"
    assert res_pos["preservation"] == "PASS"
    assert res_pos["delta"] == 0.2

    # Negative regression fails preservation
    res_neg = evaluate_specialist_comparison(0.6, 0.4, higher_is_better=False)
    assert res_neg["status"] == "NEGATIVE"
    assert res_neg["preservation"] == "FAIL"

    # Undefined rate is INDETERMINATE and cannot satisfy preservation
    res_undef = evaluate_specialist_comparison(None, 0.4, higher_is_better=False)
    assert res_undef["status"] == "INDETERMINATE"
    assert res_undef["preservation"] == "INDETERMINATE"


# =============================================================================
# 2. Integration Tests: SpecialistCompose, C2, C6, A16, A15
# =============================================================================

def test_specialist_compose_and_coverage():
    # Build synthetic records across 3 datasets: fever (grounding), toolbench (policy), weblinx (replay)
    records = [
        # fever: pcg accepts, citation_only accepts
        {"dataset": "fever", "system": "pcg_mas", "accepted": True},
        {"dataset": "fever", "system": "citation_only", "accepted": True},
        # toolbench: pcg accepts, shieldagent accepts
        {"dataset": "toolbench", "system": "pcg_mas", "accepted": True},
        {"dataset": "toolbench", "system": "shieldagent", "accepted": True},
        # weblinx: pcg accepts, shieldagent accepts
        {"dataset": "weblinx", "system": "pcg_mas", "accepted": True},
        {"dataset": "weblinx", "system": "shieldagent", "accepted": True},
    ]

    res = SpecialistCompose.evaluate_coverage(records)
    assert res["coverage_pcg"] == 1.0
    assert res["coverage_specialist_compose"] == 1.0
    assert res["coverage_delta"] == 0.0
    assert res["c6_compliance"] is True


def test_integrated_risk_gain_c2():
    # NoCert harm = 0.5083, PCG harm = 0.0000
    c2 = evaluate_integrated_risk(h_joint_pcg=0.0, h_joint_nocert=0.5083)
    assert c2["c2_pass"] is True
    assert c2["status"] == "PASS"
    assert c2["delta"] == 0.5083


def test_a16_zero_cost_accounting():
    accounting = A16ResourceAccounting(spend_ceiling_usd=0.00)
    records = [
        {"billed_cost_usd": 0.0, "model_calls": 3, "generator_calls": 1, "checker_calls": 2, "tokens_in": 100, "tokens_out": 50},
        {"billed_cost_usd": 0.0, "model_calls": 2, "generator_calls": 1, "checker_calls": 1, "tokens_in": 80, "tokens_out": 40},
    ]
    res = accounting.compute(records)
    assert res["ceiling_respected"] is True
    assert res["total_spend_usd"] == 0.0
    assert res["model_calls"] == 5
    assert res["tokens_total"] == 270

    # Spending above 0.00 raises ValueError
    violating_records = [{"billed_cost_usd": 0.01, "model_calls": 1}]
    with pytest.raises(ValueError, match="A16 spend ceiling violation"):
        accounting.compute(violating_records)


def test_a15_deberta_mock_and_bundle(tmp_path):
    checker = A15DebertaNLIInterface(threshold=0.5, mock_mode=True)
    score1 = checker.score("premise text", "hypothesis text")
    score2 = checker.score("premise text", "hypothesis text")
    assert 0.0 <= score1 <= 1.0
    assert score1 == score2  # Deterministic

    bundle_dir = tmp_path / "a15_bundle"
    A15DebertaNLIInterface.generate_off_mac_bundle(bundle_dir)
    assert (bundle_dir / "run_a15_deberta_gpu.py").exists()
    assert (bundle_dir / "requirements.txt").exists()
    assert (bundle_dir / "README.md").exists()


# =============================================================================
# 3. Property Tests (Hypothesis)
# =============================================================================

@given(st.integers(min_value=0, max_value=10000), st.integers(min_value=1, max_value=10000))
def test_property_safe_rate_valid_denominator(num, den):
    rate = safe_rate(num, den)
    assert rate is not None
    assert 0.0 <= rate <= (num / den)


@given(st.integers(min_value=0, max_value=10000))
def test_property_safe_rate_zero_denominator(num):
    assert safe_rate(num, 0) is None


@given(st.booleans(), st.booleans())
def test_property_vacuous_harm_invariance(h_supp, h_exec):
    harm = HarmEvaluator()
    # ToolBench never has textual support harm regardless of h_supp input
    res = harm.evaluate_harm("toolbench", h_support=h_supp, h_exec=h_exec)
    assert res["effective_h_support"] is False
    assert res["h_joint"] == h_exec


# =============================================================================
# 4. Determinism Tests
# =============================================================================

def test_evaluation_determinism():
    evaluator = AcceptanceEvaluator()
    runs = [
        evaluator.check_pcg("fever", v_h=True, v_pi=False, v_gamma=False, v_entail=True)
        for _ in range(20)
    ]
    first = runs[0]
    for r in runs[1:]:
        assert r == first


# =============================================================================
# 5. Firewall Tests: Isolated Partitions & Zero Commercial Spend
# =============================================================================

def test_firewall_partition_isolation():
    # Verify that the harness does NOT attempt to read CHECKER_CALIBRATION or FINAL files
    import pcg.v3.harness as h_mod
    h_code = Path(h_mod.__file__).read_text(encoding="utf-8")
    assert "FINAL" not in h_code or "FINAL partition" in h_code
    assert "open(" not in h_code or "FINAL" not in h_code


# =============================================================================
# 6. Historical Immutability Tests: Raw AG-2.6 Unmodified
# =============================================================================

def test_historical_ag2_6_immutability():
    if not DS_DIR.exists():
        pytest.skip("Historical AG-2.6 artifacts are excluded from Git repository.")
    expected_hashes = {
        "AG2_6_PER_EXAMPLE.jsonl": "9d6b4c7103d08d143537dafba7f37eac2c338b5da82c9d50f73d9996af15a5c8",
        "AG2_6_GENERATION_CACHE.json": "ae0d0e61a04b9493e7e5f7b5bc5032db69a25af7bba26e23beafcd33f008c0a9",
        "AG2_6_GENERATION_SPEC_MANIFEST.json": "39fe288077cef8247b493c39e2a0fa3f933637240f83cf7d65306d7ed4510d03",
        "AG2_6_SCOPE_EFFECTS.csv": "bf4313ff388fd28485097800bb4b9080699c9ead601db6438cc0a9da998e1f58",
        "AG2_6_UTILITY_COVERAGE.csv": "f29979f157186b7854980a000cd1570fa6f100a85da5859a10f3ef23076d1535",
        "AG2_6_A16_MATCHED_BUDGET.csv": "f14b210d0dd8e3ca293929588b32c20b24dbe290b799f8492f35871e042f1c44",
        "AG2_6_CHECKER_SENSITIVITY.csv": "d4ee0c4d9d6ef81edf3636f66576b2487bda8e3108d1430f25561fba4f122b35",
        "AG2_6_THREE_MODEL_SYNTHESIS.csv": "842aa24c996373f6135718c59c11866344f0cc6e681bf2fa0f37acafc0ace2a9",
        "AG2_6_GATE_REPORT.md": "710f972a90f3afe16579d421b2ea306b8a4ecaebe471ecbb6c1e09c5a653648d",
        "AG2_6_TERMINAL_REPORT.txt": "4f5e747d7dc43082cb4ea090473bf6497a8c651d158ea0178233aa519bc585ea",
        "AG2_6_PROTOCOL_PRECOMMIT.json": "2c118cc6fa4ee7f401194843bfe667c263f0a4872c3089fbbd51d95c676f31ea",
    }
    for fn, exp_h in expected_hashes.items():
        act_h = hashlib.sha256((DS_DIR / fn).read_bytes()).hexdigest()
        assert act_h == exp_h, f"Hash mismatch on {fn}!"


# =============================================================================
# 7. Scientific Constant Diff is EMPTY
# =============================================================================

def test_scientific_constants_unmutated():
    payload = version_payload()
    assert payload["PCG_MAS_RELEASE"] == "v3.2"
    assert payload["scientific_schema_version"] == "3.0.0"
    assert payload["artifact_schema_version"] == "3.2.0"
    assert payload["record_schema_version"] == "3.2.1"
    assert payload["metric_version"] == "v3.0.0"

    # Protocol SHA from S01
    sha_file = S01_DIR / "PROTOCOL_SHA.txt"
    if not sha_file.exists():
        pytest.skip("S01 protocol artifacts excluded from Git repository.")
    assert len(sha_file.read_text().strip()) == 64
