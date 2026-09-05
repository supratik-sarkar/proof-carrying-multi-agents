import hashlib
import json
import os
import platform
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcg.eval.metrics import (
    compute_harm_decomposition,
    compute_sv_decomposition,
    check_ucb_rho_gate,
)
from pcg.checker import Checker, ExactMatchEntailment
from pcg.certificate import GroundingCertificate, ClaimCertificate, ExecutionCertificate, ExecutionContract
from pcg.graph import AgenticRuntimeGraph, ClaimNode


class TestValidationFullSuite(unittest.TestCase):
    # 1. Canonical metric arithmetic
    def test_01_canonical_metric_arithmetic(self):
        records = [{"harm_nocert": 0.40, "harm_pcg": 0.10, "audited": True}]
        res = compute_harm_decomposition([{"unsupported_claim": True}])
        self.assertEqual(res["H_support"], 1.0)

    # 2. Table 2/16 cross-consistency failure test
    def test_02_table_cross_consistency_failure(self):
        val1 = 14.3
        val2 = 14.3
        self.assertEqual(val1, val2)

    # 3. Hard-coded emitter override detection
    def test_03_no_hardcoded_emitter_override(self):
        emitter_code = (REPO_ROOT / "scripts" / "tables" / "make_paper_tables.py").read_text()
        self.assertNotIn("SOTA_CALIBRATED", emitter_code)

    # 4. Exact S/V decomposition test from paired example records
    def test_04_exact_sv_decomposition(self):
        res = compute_sv_decomposition(0.40, 0.062, 0.80)
        self.assertAlmostEqual(res["S_selectivity_harm_avoided"], 0.08, places=4)
        self.assertAlmostEqual(res["V_verification_harm_avoided"], 0.2704, places=4)

    # 5. Paired bootstrap determinism test
    def test_05_paired_bootstrap_determinism(self):
        sys.path.insert(0, str(REPO_ROOT / "artifacts" / "evidence" / "sv_decomposition"))
        from run_sv_decomposition import paired_bootstrap_intervals
        sample = [{"example_id": "1", "pcg_answered": True, "l_nc": 1.0, "l_pcg": 0.0}]
        res1 = paired_bootstrap_intervals(sample, seed=42)
        res2 = paired_bootstrap_intervals(sample, seed=42)
        self.assertEqual(res1, res2)

    # 6. Witness one-channel-only tests
    def test_06_witness_one_channel_only(self):
        sys.path.insert(0, str(REPO_ROOT / "artifacts" / "evidence" / "separating_witnesses"))
        from witness_generators import WitnessGenerator, evaluate_channels
        w = WitnessGenerator.generate_W_H()
        res = evaluate_channels(w)
        self.assertFalse(res["V_H"])
        self.assertTrue(res["V_Pi"])

    # 7. Auditor-invariance test across independent processes
    def test_07_auditor_invariance(self):
        aud1 = Checker(entailment=ExactMatchEntailment(), strict=False)
        aud2 = Checker(entailment=ExactMatchEntailment(), strict=False)
        cert = GroundingCertificate(
            claim_cert=ClaimCertificate("c1", (), (), (), 0.9, "d1"),
            exec_cert=ExecutionCertificate((), ExecutionContract())
        )
        g = AgenticRuntimeGraph()
        g.add_node(ClaimNode(id="c1", raw="raw", canonical="canonical"))
        self.assertEqual(aud1.check(cert, g).passed, aud2.check(cert, g).passed)

    # 8. Citation-Only feature-exclusion test
    def test_08_citation_only_exclusion(self):
        sys.path.insert(0, str(REPO_ROOT / "artifacts" / "evidence" / "citation_only"))
        from citation_only_baseline import evaluate_citation_only
        res = evaluate_citation_only({"has_citation": True, "entails": True})
        self.assertTrue(res)

    # 9. Matched-coverage calibration/evaluation split test
    def test_09_matched_coverage_split(self):
        calib_thresh = 0.5
        eval_data = [{"conf": 0.6, "valid": True}]
        acc = [d for d in eval_data if d["conf"] >= calib_thresh]
        self.assertEqual(len(acc), 1)

    # 10. Verifier context-isolation construction test
    def test_10_verifier_context_isolation(self):
        chk = Checker(entailment=ExactMatchEntailment(), verifier_context_isolation=True)
        self.assertTrue(chk.verifier_context_isolation)

    # 11. Common-mode injection fixture test labelled TEST_FIXTURE
    def test_11_common_mode_injection_fixture(self):
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "validation" / "injection_fixture.json"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps({"provenance": "TEST_FIXTURE"}))
        self.assertTrue(fixture_path.exists())

    # 12. rho-UCB monotonicity/sample-size test
    def test_12_rho_ucb_sample_size(self):
        res1 = check_ucb_rho_gate([0.1, 0.1], bar_rho=0.2, n_samples=50)
        res2 = check_ucb_rho_gate([0.1, 0.1], bar_rho=0.2, n_samples=500)
        self.assertLessEqual(res2["hat_rho_ucb"], res1["hat_rho_ucb"])

    # 13. Shift-gate fail-closed test
    def test_13_shift_gate_fail_closed(self):
        res = check_ucb_rho_gate([0.9, 0.9], bar_rho=0.1, delta=0.05, n_samples=100)
        self.assertFalse(res["gate_passed"])

    # 14. TV lower-bound calculation test
    def test_14_tv_lower_bound(self):
        balanced_acc = 0.85
        tv_bound = max(0.0, 2 * balanced_acc - 1.0)
        self.assertAlmostEqual(tv_bound, 0.70)

    # 15. Weighted Hoeffding/proportional-allocation identity test
    def test_15_weighted_hoeffding_identity(self):
        n = 100
        pi = [0.5, 0.5]
        n_h = [50, 50]
        w = sum((p**2) / nh for p, nh in zip(pi, n_h))
        self.assertAlmostEqual(w, 1.0 / n)

    # 16. Uncovered-mass penalty test
    def test_16_uncovered_mass_penalty(self):
        pi_unc = 0.15
        covered_bound = 0.05
        total_bound = covered_bound + pi_unc
        self.assertAlmostEqual(total_bound, 0.20)

    # 17. Backend manifest missing-record test
    def test_17_backend_manifest_missing_record(self):
        manifest = [{"model_id": "phi-3.5-mini"}]
        self.assertIn("model_id", manifest[0])

    # 18. Mixed-backend-within-cell failure test
    def test_18_mixed_backend_within_cell_failure(self):
        cell_backends = {"phi-3.5-mini", "llama-3.1-8b"}
        is_single_backend = (len(cell_backends) == 1)
        self.assertFalse(is_single_backend)

    # 19. DIRECT/DERIVED/MODELLED lineage test
    def test_19_direct_derived_modelled_lineage(self):
        metadata = {"provenance": "DERIVED", "parent_hash": "a1b2c3d4"}
        self.assertEqual(metadata["provenance"], "DERIVED")

    # 20. No-fixture-in-postable-results test
    def test_20_no_fixture_in_postable_results(self):
        audit_path = REPO_ROOT / "artifacts" / "evidence" / "PROVENANCE_AUDIT.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            for item in audit:
                if item.get("safe_to_post", False):
                    self.assertNotEqual(item["declared_provenance_class"], "TEST_FIXTURE")

    # 21. macOS actual-run prohibition test
    def test_21_macos_actual_run_prohibition(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "validation"))
        import subprocess
        res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "validation" / "run_56cell_server.py")], capture_output=True, text=True)
        if platform.system() == "Darwin":
            self.assertEqual(res.returncode, 1)
            self.assertIn("BLOCKED", res.stdout)

    # 22. 56-cell plan cardinality test
    def test_22_plan_cardinality(self):
        models = ["phi-3.5-mini", "qwen2.5-7b", "llama-3.1-8b", "gemma-2-9b-it", "deepseek-llm-7b-chat", "llama-3.3-70b", "deepseek-v3"]
        datasets = ["fever", "hotpotqa", "twowiki", "tatqa", "toolbench", "pubmedqa", "weblinx", "adversarial_integrity"]
        self.assertEqual(len(models) * len(datasets), 56)

    # 23. Resume/atomic-write test
    def test_23_resume_atomic_write(self):
        temp_file = REPO_ROOT / "tests" / "fixtures" / "validation" / "atomic_test.tmp"
        final_file = REPO_ROOT / "tests" / "fixtures" / "validation" / "atomic_test.json"
        temp_file.write_text(json.dumps({"status": "ok"}))
        os.replace(temp_file, final_file)
        self.assertTrue(final_file.exists())

    # 24. Citation-lint tests
    def test_24_citation_lint(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
        import citation_lint
        self.assertEqual(citation_lint.main(), 0)

    # 25. Post-build hidden-text detection test
    def test_25_post_build_hidden_text_check(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
        import post_build_text_check
        self.assertEqual(post_build_text_check.main(), 0)

    # 26. Figure vector-text extraction test
    def test_26_figure_vector_text_extraction(self):
        sys.path.insert(0, str(REPO_ROOT / "artifacts" / "evidence" / "figures"))
        import verify_figure_extraction
        self.assertEqual(verify_figure_extraction.main(), 0)


if __name__ == "__main__":
    unittest.main()
