import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcg.checker import Checker, ExactMatchEntailment
from pcg.certificate import GroundingCertificate, ClaimCertificate, ExecutionCertificate, ExecutionContract
from pcg.graph import AgenticRuntimeGraph, ClaimNode


class TestAuditorInvariance(unittest.TestCase):
    def test_independent_auditor_processes_reach_identical_verdicts(self):
        """Instantiates two independent auditor processes and asserts identical channel & composite verdicts."""
        auditor_1 = Checker(
            entailment=ExactMatchEntailment(),
            strict=False,
            verifier_context_isolation=True
        )

        auditor_2 = Checker(
            entailment=ExactMatchEntailment(),
            strict=False,
            verifier_context_isolation=True
        )

        cert = GroundingCertificate(
            claim_cert=ClaimCertificate(
                claim_id="c1",
                evidence_ids=(),
                evidence_digests=(),
                pipeline=(),
                confidence=0.9,
                replay_output_digest="digest_1"
            ),
            exec_cert=ExecutionCertificate(
                pipeline=(),
                contract=ExecutionContract(allowed_tools=frozenset({"search", "calculator"}))
            )
        )
        graph = AgenticRuntimeGraph()
        claim_node = ClaimNode(id="c1", raw="Paris is the capital of France.", canonical="Paris is the capital of France.")
        graph.add_node(claim_node)

        res1 = auditor_1.check(cert, graph)
        res2 = auditor_2.check(cert, graph)

        self.assertEqual(res1.passed, res2.passed)
        self.assertEqual(res1.V_H, res2.V_H)
        self.assertEqual(res1.V_Pi, res2.V_Pi)
        self.assertEqual(res1.V_Gamma, res2.V_Gamma)
        self.assertEqual(res1.V_entail, res2.V_entail)
        self.assertEqual(res1.reasons, res2.reasons)


if __name__ == "__main__":
    unittest.main()
