"""PCG-MAS v3.4 Task Obligations Adapter.

Implements Requirement B:
- Obligations O(x) = {o_1, ..., o_m} derived deterministically from task contract
- Fixed denominator: model-emitted claims cannot shrink the denominator
- Stable IDs and content hashes
- Support for critical vs standard obligations
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass(frozen=True)
class TaskObligation:
    obligation_id: str
    dataset: str
    obligation_type: str
    description: str
    weight: float = 1.0
    is_critical: bool = True
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            h = hashlib.sha256(f"{self.dataset}:{self.obligation_id}:{self.obligation_type}:{self.description}".encode("utf-8")).hexdigest()
            object.__setattr__(self, "content_hash", h)

@dataclass
class AtomicClaim:
    claim_id: str
    text: str
    mapped_obligation_id: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            h = hashlib.sha256(f"{self.claim_id}:{self.text}".encode("utf-8")).hexdigest()
            self.content_hash = h

def derive_task_obligations(dataset: str, example_id: str, question: str, meta: Optional[Dict[str, Any]] = None) -> List[TaskObligation]:
    """Derives deterministic task obligations O(x) from task structure alone."""
    meta = meta or {}
    obligations = []
    
    if dataset == "fever":
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_verdict",
            dataset="fever",
            obligation_type="verdict_decision",
            description="Commitment to a discrete claim verdict (SUPPORTS, REFUTES, or NOT ENOUGH INFO)",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_grounding",
            dataset="fever",
            obligation_type="claim_grounding",
            description="Verification of the core proposition against retrieved evidence",
            weight=1.0,
            is_critical=True
        ))
    elif dataset in ("hotpotqa", "twowiki"):
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_answer",
            dataset=dataset,
            obligation_type="answer_commitment",
            description="Direct answer addressing the user query",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_hop1",
            dataset=dataset,
            obligation_type="primary_relation_grounding",
            description="Grounding of primary entity relation against cited passage",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_hop2",
            dataset=dataset,
            obligation_type="multihop_bridge_grounding",
            description="Grounding of multihop reasoning link between documents",
            weight=1.0,
            is_critical=False
        ))
    elif dataset == "pubmedqa":
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_decision",
            dataset="pubmedqa",
            obligation_type="clinical_decision",
            description="Categorical clinical answer (yes, no, or maybe)",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_rationale",
            dataset="pubmedqa",
            obligation_type="biomedical_rationale_grounding",
            description="Medical rationale grounded in biomedical abstract",
            weight=1.0,
            is_critical=True
        ))
    elif dataset == "tatqa":
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_value",
            dataset="tatqa",
            obligation_type="numerical_value_commitment",
            description="Calculated financial or numerical value",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_table_ground",
            dataset="tatqa",
            obligation_type="tabular_text_grounding",
            description="Grounding against relevant table cells and narrative context",
            weight=1.0,
            is_critical=True
        ))
    elif dataset == "toolbench":
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_syntax",
            dataset="toolbench",
            obligation_type="tool_syntax_schema",
            description="Syntactically valid tool call matching function schema",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_policy",
            dataset="toolbench",
            obligation_type="pre_execution_policy",
            description="Execution conforms to sandbox security and authorization policy",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_replay",
            dataset="toolbench",
            obligation_type="deterministic_trace_replay",
            description="Deterministic execution trajectory verifiable via replay witness",
            weight=1.0,
            is_critical=True
        ))
    elif dataset == "weblinx":
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_action",
            dataset="weblinx",
            obligation_type="browser_action_schema",
            description="Valid web interaction action targeting valid DOM element",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_policy",
            dataset="weblinx",
            obligation_type="web_policy_compliance",
            description="Action satisfies boundary safety and site policy rules",
            weight=1.0,
            is_critical=True
        ))
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_replay",
            dataset="weblinx",
            obligation_type="dom_state_replay",
            description="DOM state transition matches expected trajectory under replay",
            weight=1.0,
            is_critical=True
        ))
    else:
        obligations.append(TaskObligation(
            obligation_id=f"{example_id}_obl_generic",
            dataset=dataset,
            obligation_type="generic_grounding",
            description="Default task correctness obligation",
            weight=1.0,
            is_critical=True
        ))
    return obligations

def compute_obligation_coverage(
    obligations: List[TaskObligation],
    certified_obligation_ids: set[str]
) -> float:
    """Computes obligation coverage C_obl = sum(w_j * 1[o_j certified]) / sum(w_j).
    
    CRITICAL: The denominator is the fixed sum of weights over all task obligations O(x).
    Emitting fewer claims can never reduce this denominator.
    """
    if not obligations:
        return 0.0
    total_weight = sum(o.weight for o in obligations)
    if total_weight <= 0:
        return 0.0
    certified_weight = sum(o.weight for o in obligations if o.obligation_id in certified_obligation_ids)
    return certified_weight / total_weight

def verify_critical_obligations(
    obligations: List[TaskObligation],
    certified_obligation_ids: set[str]
) -> bool:
    """Verifies whether ALL critical obligations are certified.
    
    One highly supported non-critical claim can NEVER rescue an unsupported critical obligation.
    """
    for o in obligations:
        if o.is_critical and o.obligation_id not in certified_obligation_ids:
            return False
    return True
