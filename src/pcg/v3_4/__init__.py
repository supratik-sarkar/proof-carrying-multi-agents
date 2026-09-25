"""PCG-MAS v3.4 Experimental Package."""
from pcg.v3_4.states import RawVerifierState, effective_binary_bit, compute_operational_acceptance
from pcg.v3_4.obligations import TaskObligation, AtomicClaim, derive_task_obligations, compute_obligation_coverage, verify_critical_obligations
from pcg.v3_4.semantic_gate import NLIWindowScore, SemanticVerificationResult, evaluate_vector_semantic_gate
from pcg.v3_4.replay_policy import ActionTraceStep, check_semantic_replay_equivalence, check_policy_compliance, evaluate_replay_and_policy
from pcg.v3_4.mce import MCEResult, compute_mce_post_pass
from pcg.v3_4.two_stage import TwoStageExecutionResult, execute_two_stage_certificate
from pcg.v3_4.harm_utility import safe_divide, evaluate_grounding_independent_harm_and_success, evaluate_interactive_independent_harm_and_success, compute_population_metrics
