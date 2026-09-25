"""Production task input obligation engine for PCG-MAS v3.5.

Enforces:
- Deterministic task input adapters O(x) for all 7 datasets.
- Reference independence (O1): gold / reference / label fields have ZERO effect on O(x).
- Obligation-specific hypotheses (O2): no monolithic whole-answer reuse across distinct obligations.
- Exactly K0=3 evidence slots per obligation (O3): missing slots are designated ABSENCE_SLOT and trigger 0 NLI calls.
- Generated answer mutation cannot change obligation set / denominator (O4).
- Critical vs supplementary mapping is deterministic and hashed (O5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

K0_SLOTS = 3


@dataclass(frozen=True)
class ObligationDefinition:
    """Immutable definition of an obligation derived strictly from task input x."""

    obligation_id: str
    dataset: str
    example_id: str
    obligation_type: str
    is_critical: bool
    template_name: str
    description: str


@dataclass(frozen=True)
class BoundHypothesis:
    """An obligation bound to a specific hypothesis string for a candidate."""

    obligation_id: str
    obligation_type: str
    is_critical: bool
    hypothesis_text: str
    provenance: str


@dataclass
class EvidenceSlot:
    """An evidence slot: either REAL with text, or ABSENCE_SLOT."""

    slot_index: int
    slot_type: str  # "REAL" or "ABSENCE_SLOT"
    text: Optional[str] = None


@dataclass
class ScorerTraceRecord:
    """Instrumentation record for an NLI verification call."""

    obligation_id: str
    obligation_type: str
    is_critical: bool
    slot_index: int
    slot_type: str
    evidence_text: Optional[str]
    hypothesis_text: str
    p_contradiction: float
    p_entailment: float
    p_neutral: float


class TaskInputObligationAdapter:
    """Deterministic task input adapter mapping x -> Set[ObligationDefinition]."""

    SUPPORTED_DATASETS = {
        "fever",
        "hotpotqa",
        "pubmedqa",
        "tatqa",
        "twowiki",
        "toolbench",
        "weblinx",
    }

    @staticmethod
    def _clean_input(x: Dict[str, Any]) -> Dict[str, Any]:
        """Strip any reference/gold/evaluator fields to enforce reference independence."""
        forbidden_keys = {
            "gold_answer",
            "reference",
            "reference_answer",
            "ground_truth",
            "label",
            "evaluator_labels",
            "harm",
            "success",
            "correctness",
            "target",
        }
        return {k: v for k, v in x.items() if k not in forbidden_keys}

    @classmethod
    def get_obligations(
        cls,
        dataset: str,
        example_id: str,
        task_input: Dict[str, Any],
    ) -> List[ObligationDefinition]:
        """Derive deterministic obligations O(x) strictly from task input x."""
        clean_x = cls._clean_input(task_input)
        d_lower = dataset.lower()
        if d_lower not in cls.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset for v3.5 obligations: {dataset}")

        obls: List[ObligationDefinition] = []

        if d_lower == "fever":
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_0_verdict",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="verdict_decision",
                    is_critical=True,
                    template_name="fever_verdict",
                    description="Claim verification verdict commitment",
                )
            )
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_1_grounding",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="claim_grounding",
                    is_critical=False,
                    template_name="fever_grounding",
                    description="Evidence grounding of claim proposition",
                )
            )

        elif d_lower == "hotpotqa":
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_0_answer",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="factual_answer",
                    is_critical=True,
                    template_name="hotpot_answer",
                    description="Direct multi-hop factual answer commitment",
                )
            )
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_1_supporting",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="supporting_facts",
                    is_critical=False,
                    template_name="hotpot_supporting",
                    description="Supporting facts attribution",
                )
            )

        elif d_lower == "pubmedqa":
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_0_verdict",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="biomedical_verdict",
                    is_critical=True,
                    template_name="pubmed_verdict",
                    description="Biomedical research verdict commitment",
                )
            )
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_1_rationale",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="rationale_grounding",
                    is_critical=False,
                    template_name="pubmed_rationale",
                    description="Biomedical abstract rationale grounding",
                )
            )

        elif d_lower == "tatqa":
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_0_calculation",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="arithmetic_calculation",
                    is_critical=True,
                    template_name="tatqa_calc",
                    description="Financial arithmetic derivation commitment",
                )
            )
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_1_scale",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="scale_unit_consistency",
                    is_critical=False,
                    template_name="tatqa_scale",
                    description="Financial unit and scale consistency",
                )
            )

        elif d_lower == "twowiki":
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_0_hop1",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="hop1_grounding",
                    is_critical=True,
                    template_name="twowiki_hop1",
                    description="First reasoning hop entity grounding",
                )
            )
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_1_hop2",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="hop2_grounding",
                    is_critical=False,
                    template_name="twowiki_hop2",
                    description="Second reasoning hop relation grounding",
                )
            )

        elif d_lower == "toolbench":
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_0_syntax",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="tool_syntax_schema",
                    is_critical=True,
                    template_name="toolbench_syntax",
                    description="Tool call syntax and parameter validity",
                )
            )
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_1_policy",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="pre_execution_policy",
                    is_critical=True,
                    template_name="toolbench_policy",
                    description="Pre-execution policy authorization compliance",
                )
            )

        elif d_lower == "weblinx":
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_0_action",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="browser_action_schema",
                    is_critical=True,
                    template_name="weblinx_action",
                    description="Browser action schema and target element validity",
                )
            )
            obls.append(
                ObligationDefinition(
                    obligation_id=f"{dataset}:{example_id}:ob_1_policy",
                    dataset=dataset,
                    example_id=example_id,
                    obligation_type="web_policy_compliance",
                    is_critical=True,
                    template_name="weblinx_policy",
                    description="Web interaction policy and navigational safety",
                )
            )

        return obls

    @classmethod
    def hash_obligations(cls, obls: List[ObligationDefinition]) -> str:
        """Compute deterministic SHA256 of the obligation list."""
        repr_str = json.dumps(
            [
                {
                    "obligation_id": o.obligation_id,
                    "dataset": o.dataset,
                    "example_id": o.example_id,
                    "obligation_type": o.obligation_type,
                    "is_critical": o.is_critical,
                    "template_name": o.template_name,
                }
                for o in obls
            ],
            sort_keys=True,
        )
        return hashlib.sha256(repr_str.encode("utf-8")).hexdigest()


def derive_obligation_hypotheses(
    obls: List[ObligationDefinition],
    prompt_text: str,
    output_text: str,
) -> List[BoundHypothesis]:
    """Derive distinct obligation-specific hypotheses.

    Enforces that distinct obligations NEVER receive the exact same hypothesis text.
    """
    results: List[BoundHypothesis] = []

    for o in obls:
        otype = o.obligation_type
        if otype in ("verdict_decision", "biomedical_verdict"):
            hyp_text = f"Claim verdict commitment: {output_text.strip()}"
        elif otype in ("claim_grounding", "rationale_grounding"):
            hyp_text = f"Claim proposition: {prompt_text.strip()}"
        elif otype in ("factual_answer", "arithmetic_calculation"):
            hyp_text = f"Question: {prompt_text.strip()} Answer commitment: {output_text.strip()}"
        elif otype in ("supporting_facts", "scale_unit_consistency"):
            hyp_text = f"Supporting attribution: {prompt_text.strip()} -> {output_text.strip()[:100]}"
        elif otype == "hop1_grounding":
            hyp_text = f"Reasoning step 1: {prompt_text.strip()}"
        elif otype == "hop2_grounding":
            hyp_text = f"Reasoning step 2 connecting to: {output_text.strip()}"
        elif otype in ("tool_syntax_schema", "browser_action_schema"):
            hyp_text = f"Valid execution syntax for action: {output_text.strip()[:100]}"
        elif otype in ("pre_execution_policy", "web_policy_compliance"):
            hyp_text = f"Policy authorization compliance for action: {output_text.strip()[:100]}"
        else:
            hyp_text = f"Obligation [{otype}]: {prompt_text.strip()} -> {output_text.strip()}"

        results.append(
            BoundHypothesis(
                obligation_id=o.obligation_id,
                obligation_type=o.obligation_type,
                is_critical=o.is_critical,
                hypothesis_text=hyp_text,
                provenance="deterministic_obligation_specific_template",
            )
        )

    # Invariant: if multiple obligations, hypotheses must be distinct
    if len(results) > 1:
        hyps = {b.hypothesis_text for b in results}
        if len(hyps) == 1:
            raise ValueError(
                f"Obligation invariant violation: all {len(results)} obligations "
                "received identical hypothesis text!"
            )

    return results


def build_evidence_slots(
    raw_evidence_windows: List[str],
    k0: int = K0_SLOTS,
) -> List[EvidenceSlot]:
    """Construct exactly k0 evidence slots.

    If fewer than k0 real windows exist, pad with ABSENCE_SLOT.
    """
    slots: List[EvidenceSlot] = []
    real_windows = [w for w in raw_evidence_windows if w and w.strip()]

    for i in range(k0):
        if i < len(real_windows):
            slots.append(
                EvidenceSlot(
                    slot_index=i,
                    slot_type="REAL",
                    text=real_windows[i],
                )
            )
        else:
            slots.append(
                EvidenceSlot(
                    slot_index=i,
                    slot_type="ABSENCE_SLOT",
                    text=None,
                )
            )
    return slots


def evaluate_obligations_with_trace(
    hypotheses: List[BoundHypothesis],
    evidence_slots: List[EvidenceSlot],
    scorer_fn: Callable[[str, str], Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], List[ScorerTraceRecord]]:
    """Evaluate hypotheses against evidence slots using scorer_fn.

    CRITICAL RULES:
    1. ABSENCE_SLOT triggers ZERO scorer_fn / NLI calls.
    2. S_i = max p_E over real evidence slots (or 0.0 if 0 real slots).
    3. K_i = max p_C over real evidence slots (or 0.0 if 0 real slots).
    4. Records full trace for verification.
    """
    trace: List[ScorerTraceRecord] = []
    obligation_scores: Dict[str, Dict[str, float]] = {}

    for h in hypotheses:
        s_vals: List[float] = []
        k_vals: List[float] = []

        for slot in evidence_slots:
            if slot.slot_type == "ABSENCE_SLOT":
                # ABSENCE_SLOT must NEVER trigger NLI calls
                trace.append(
                    ScorerTraceRecord(
                        obligation_id=h.obligation_id,
                        obligation_type=h.obligation_type,
                        is_critical=h.is_critical,
                        slot_index=slot.slot_index,
                        slot_type="ABSENCE_SLOT",
                        evidence_text=None,
                        hypothesis_text=h.hypothesis_text,
                        p_contradiction=0.0,
                        p_entailment=0.0,
                        p_neutral=0.0,
                    )
                )
                continue

            # REAL slot: invoke scorer_fn
            scores = scorer_fn(slot.text, h.hypothesis_text)
            pe = scores["p_entailment"]
            pc = scores["p_contradiction"]
            pn = scores.get("p_neutral", 0.0)

            s_vals.append(pe)
            k_vals.append(pc)

            trace.append(
                ScorerTraceRecord(
                    obligation_id=h.obligation_id,
                    obligation_type=h.obligation_type,
                    is_critical=h.is_critical,
                    slot_index=slot.slot_index,
                    slot_type="REAL",
                    evidence_text=slot.text,
                    hypothesis_text=h.hypothesis_text,
                    p_contradiction=pc,
                    p_entailment=pe,
                    p_neutral=pn,
                )
            )

        # S_i = max p_E over real slots, K_i = max p_C over real slots
        s_i = max(s_vals) if s_vals else 0.0
        k_i = max(k_vals) if k_vals else 0.0

        obligation_scores[h.obligation_id] = {
            "S_i": float(s_i),
            "K_i": float(k_i),
            "is_critical": 1.0 if h.is_critical else 0.0,
        }

    return obligation_scores, trace


def verify_obligations_domain() -> Dict[str, Any]:
    """Production verification callable for obligations domain."""
    slots = build_evidence_slots(["doc_0"])
    return {"domain": "obligations", "slot_count": len(slots)}


