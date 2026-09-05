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
from pcg.checker import Checker


class TestValidationArchitecture(unittest.TestCase):
    def test_harm_decomposition(self):
        records = [
            {"unsupported_claim": True, "entailment_fail": False, "disallowed_tool": False},
            {"unsupported_claim": False, "entailment_fail": False, "disallowed_tool": True},
            {"unsupported_claim": False, "entailment_fail": False, "disallowed_tool": False},
        ]
        res = compute_harm_decomposition(records)
        self.assertEqual(res["H_support"], 0.3333)
        self.assertEqual(res["H_exec"], 0.3333)
        self.assertEqual(res["composite_harm"], 0.6667)

    def test_sv_decomposition(self):
        # harm_nocert = 0.40, harm_pcg = 0.062, accept_rate = 0.80
        res = compute_sv_decomposition(0.40, 0.062, 0.80)
        self.assertEqual(res["S_selectivity_harm_avoided"], 0.08)
        self.assertEqual(res["V_verification_harm_avoided"], 0.2704)
        self.assertEqual(res["total_harm_reduction"], 0.3504)

    def test_ucb_rho_gate(self):
        realized_rhos = [0.10, 0.12, 0.11, 0.09]
        gate_res = check_ucb_rho_gate(realized_rhos, bar_rho=0.25, delta=0.05)
        self.assertTrue(gate_res["gate_passed"])
        self.assertLessEqual(gate_res["hat_rho_ucb"], 0.30)

    def test_checker_isolation_and_noprune(self):
        class MockEntailment:
            def check(self, y, c):
                return True

        chk = Checker(entailment=MockEntailment(), verifier_context_isolation=True, noprune_mode=False)
        self.assertTrue(chk.verifier_context_isolation)
        self.assertFalse(chk.noprune_mode)


if __name__ == "__main__":
    unittest.main()
