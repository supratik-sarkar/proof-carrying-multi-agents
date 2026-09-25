"""PCG-MAS v3.4R Obligation-Specific Semantic Verification Engine.

Enforces:
- Obligation-specific hypothesis derivation and binding
- Call trace instrumentation recording (evidence, hypothesis) pairs
- Strict rejection of monolithic whole-answer reuse across distinct obligations
- Invariance to obligation ordering
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from pcg.v3_4r.candidate import RuntimeCandidate


@dataclass
class ObligationHypothesis:
    """Obligation with its own explicitly bound hypothesis text."""

    obligation_id: str
    obligation_type: str
    hypothesis_text: str
    criticality: bool
    provenance: str


class ScorerCallTrace:
    """Instruments and records all calls sent to the semantic scorer."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record_call(
        self,
        obligation_id: str,
        obligation_type: str,
        evidence_text: str,
        hypothesis_text: str,
        pe: float,
        pc: float,
        pn: float,
    ) -> None:
        self.records.append(
            {
                "obligation_id": obligation_id,
                "obligation_type": obligation_type,
                "evidence_text": evidence_text,
                "hypothesis_text": hypothesis_text,
                "pe": pe,
                "pc": pc,
                "pn": pn,
            }
        )

    def verify_distinct_hypotheses_if_multiple(self) -> bool:
        """Verifies that multiple obligations did not all receive identical hypotheses."""
        if len(self.records) <= 1:
            return True
        unique_obls = {r["obligation_id"] for r in self.records}
        if len(unique_obls) > 1:
            unique_hyps = {r["hypothesis_text"] for r in self.records}
            if len(unique_hyps) == 1:
                return False
        return True


def derive_obligation_hypotheses(
    cand: RuntimeCandidate,
) -> List[ObligationHypothesis]:
    """Derives stable obligation-specific hypotheses for each obligation in cand."""
    results: List[ObligationHypothesis] = []
    num_obls = len(cand.obligations)

    for idx, obl in enumerate(cand.obligations):
        oid = obl["obligation_id"]
        otype = obl["obligation_type"]
        crit = obl.get("is_critical", True)

        # If explicit hypothesis text was already bound
        if "hypothesis_text" in obl and obl["hypothesis_text"].strip():
            hyp_text = obl["hypothesis_text"].strip()
            provenance = "explicit_manifest_binding"
        else:
            # Obligation-specific runtime derivation
            if otype in ("verdict_decision", "verdict"):
                hyp_text = f"Claim: {cand.prompt.strip()} Verdict: {cand.candidate_answer.strip()}"
                provenance = "runtime_verdict_obligation_derivation"
            elif otype in ("claim_grounding", "grounding", "core_proposition"):
                hyp_text = f"Claim proposition: {cand.prompt.strip()}"
                provenance = "runtime_grounding_obligation_derivation"
            elif otype in ("factual_answer", "answer_commitment"):
                hyp_text = f"Question: {cand.prompt.strip()} Answer: {cand.candidate_answer.strip()}"
                provenance = "runtime_answer_obligation_derivation"
            elif otype in ("evidence_binding", "rationale_grounding"):
                hyp_text = f"Supporting evidence grounds answer: {cand.candidate_answer.strip()}"
                provenance = "runtime_evidence_binding_derivation"
            elif otype in ("tool_syntax_schema", "browser_action_schema"):
                hyp_text = f"Valid execution syntax for action: {cand.candidate_answer[:100]}"
                provenance = "runtime_syntax_obligation_derivation"
            elif otype in ("pre_execution_policy", "web_policy_compliance"):
                hyp_text = f"Policy authorization compliance for: {cand.candidate_answer[:100]}"
                provenance = "runtime_policy_obligation_derivation"
            elif otype in ("deterministic_trace_replay", "dom_state_replay"):
                hyp_text = f"Replay state fidelity for: {cand.candidate_answer[:100]}"
                provenance = "runtime_replay_obligation_derivation"
            else:
                hyp_text = f"Obligation [{otype}]: {cand.prompt.strip()} -> {cand.candidate_answer.strip()}"
                provenance = "runtime_typed_obligation_derivation"

        results.append(
            ObligationHypothesis(
                obligation_id=oid,
                obligation_type=otype,
                hypothesis_text=hyp_text,
                criticality=crit,
                provenance=provenance,
            )
        )

    # Invariant: If more than 1 obligation, they must not have identical hypotheses
    if num_obls > 1:
        hyps = {oh.hypothesis_text for oh in results}
        if len(hyps) == 1:
            raise ValueError(
                "OBLIGATION INVARIANT VIOLATION: All distinct obligations received the exact same hypothesis text!"
            )

    return results


def score_obligations_with_trace(
    cand: RuntimeCandidate,
    scorer_fn: Callable[[str, str], Tuple[float, float, float]],
    tau_cov: float = 1.0,
    tau_e: float = 0.05,
    tau_c: float = 0.30,
) -> Tuple[str, Dict[str, Any], ScorerCallTrace]:
    """Scores candidate obligations against evidence windows with instrumentation.

    scorer_fn: (premise/window, hypothesis) -> (pe, pc, pn)
    """
    hypotheses = derive_obligation_hypotheses(cand)
    call_trace = ScorerCallTrace()

    per_obligation_status = {}
    covered_critical = 0
    total_critical = 0

    for oh in hypotheses:
        if oh.criticality:
            total_critical += 1

        window_scores = []
        max_pe = 0.0
        max_pc = 0.0

        for w in cand.windows:
            pe, pc, pn = scorer_fn(w, oh.hypothesis_text)
            call_trace.record_call(
                oh.obligation_id,
                oh.obligation_type,
                w,
                oh.hypothesis_text,
                pe,
                pc,
                pn,
            )
            window_scores.append({"pe": pe, "pc": pc, "pn": pn})
            if pe > max_pe:
                max_pe = pe
            if pc > max_pc:
                max_pc = pc

        # Obligation check
        is_supported = max_pe >= tau_e and max_pc <= tau_c
        if oh.criticality and is_supported:
            covered_critical += 1

        per_obligation_status[oh.obligation_id] = {
            "supported": is_supported,
            "max_pe": max_pe,
            "max_pc": max_pc,
            "is_critical": oh.criticality,
        }

    cov_ratio = (
        (covered_critical / total_critical) if total_critical > 0 else 1.0
    )
    overall_state = "PASS" if cov_ratio >= tau_cov else "FAIL"

    return (
        overall_state,
        {
            "covered_critical": covered_critical,
            "total_critical": total_critical,
            "coverage_ratio": cov_ratio,
            "per_obligation": per_obligation_status,
        },
        call_trace,
    )


def run_obligation_kat() -> Dict[str, Any]:
    """Known-Answer Test for obligation-specific verification.

    Tests:
    1. Canary: Obligation A supported, Obligation B contradicted, Obligation C neutral.
    2. Discrimination across obligations is correct.
    3. Obligation order invariance: permuting obligation list yields identical per-obligation results.
    4. Call trace instrumentation accurately records all (evidence, hypothesis) pairs.
    5. Rejection of identical hypothesis reuse across multiple obligations.
    """
    evidence = "Apples are sweet edible fruits produced by an apple tree."

    def mock_scorer(premise: str, hyp: str) -> Tuple[float, float, float]:
        if "fruit" in hyp.lower():
            return 0.95, 0.01, 0.04  # High Entailment
        elif "iron" in hyp.lower():
            return 0.01, 0.95, 0.04  # High Contradiction
        else:
            return 0.05, 0.05, 0.90  # Neutral

    cand_ordered = RuntimeCandidate(
        candidate_id="kat_obl_cand",
        model="test_model",
        dataset="test_ds",
        example_id="kat_ex",
        request_hash="h_req",
        response_hash="h_resp",
        prompt="Describe apple properties",
        candidate_answer="Apples are fruit",
        windows=[evidence],
        evidence_hashes=["h_ev"],
        obligations=[
            {
                "obligation_id": "obl_A",
                "obligation_type": "fruit_nature",
                "hypothesis_text": "Apples are fruit",
                "is_critical": True,
            },
            {
                "obligation_id": "obl_B",
                "obligation_type": "metal_nature",
                "hypothesis_text": "Apples are made of iron",
                "is_critical": True,
            },
            {
                "obligation_id": "obl_C",
                "obligation_type": "astronomy_nature",
                "hypothesis_text": "The moon orbits the earth",
                "is_critical": False,
            },
        ],
        resource_metrics={"generative_calls": 1},
    )

    # 1. Evaluate ordered
    st1, det1, trace1 = score_obligations_with_trace(
        cand_ordered, mock_scorer, tau_cov=0.5, tau_e=0.1, tau_c=0.2
    )

    # Verify canary discrimination
    a_is_supported = det1["per_obligation"]["obl_A"]["supported"] is True
    b_is_supported = det1["per_obligation"]["obl_B"]["supported"] is True
    c_is_supported = det1["per_obligation"]["obl_C"]["supported"] is True
    # A is supported, B (contradicted) is not supported, C (neutral) is not supported
    canary_ok = a_is_supported and (not b_is_supported) and (not c_is_supported)

    # 2. Evaluate reversed order
    cand_reversed = RuntimeCandidate(
        candidate_id="kat_obl_cand_rev",
        model="test_model",
        dataset="test_ds",
        example_id="kat_ex",
        request_hash="h_req",
        response_hash="h_resp",
        prompt="Describe apple properties",
        candidate_answer="Apples are fruit",
        windows=[evidence],
        evidence_hashes=["h_ev"],
        obligations=[
            cand_ordered.obligations[2],
            cand_ordered.obligations[1],
            cand_ordered.obligations[0],
        ],
        resource_metrics={"generative_calls": 1},
    )
    st2, det2, trace2 = score_obligations_with_trace(
        cand_reversed, mock_scorer, tau_cov=0.5, tau_e=0.1, tau_c=0.2
    )

    # Verify exact per-obligation outcome invariance
    invariance_ok = True
    for oid in ["obl_A", "obl_B", "obl_C"]:
        if (
            det1["per_obligation"][oid]["supported"]
            != det2["per_obligation"][oid]["supported"]
        ):
            invariance_ok = False
        if (
            det1["per_obligation"][oid]["max_pe"]
            != det2["per_obligation"][oid]["max_pe"]
        ):
            invariance_ok = False

    # 3. Call trace instrumentation
    trace_ok = (
        len(trace1.records) == 3
        and trace1.verify_distinct_hypotheses_if_multiple()
    )

    # 4. Rejection of identical hypothesis reuse
    identical_reuse_rejected = False
    try:
        cand_invalid = RuntimeCandidate(
            candidate_id="kat_obl_invalid",
            model="test_model",
            dataset="test_ds",
            example_id="kat_ex",
            request_hash="h_req",
            response_hash="h_resp",
            prompt="Prompt",
            candidate_answer="Same answer",
            windows=[evidence],
            evidence_hashes=["h_ev"],
            obligations=[
                {
                    "obligation_id": "obl_1",
                    "obligation_type": "type1",
                    "hypothesis_text": "Same hypothesis text",
                    "is_critical": True,
                },
                {
                    "obligation_id": "obl_2",
                    "obligation_type": "type2",
                    "hypothesis_text": "Same hypothesis text",
                    "is_critical": True,
                },
            ],
            resource_metrics={"generative_calls": 1},
        )
        derive_obligation_hypotheses(cand_invalid)
    except ValueError:
        identical_reuse_rejected = True

    passed = canary_ok and invariance_ok and trace_ok and identical_reuse_rejected
    return {
        "status": "PASS" if passed else "FAIL",
        "canary_discrimination": canary_ok,
        "order_invariance": invariance_ok,
        "call_trace_recorded": trace_ok,
        "identical_hypothesis_rejected": identical_reuse_rejected,
        "passed": passed,
    }
