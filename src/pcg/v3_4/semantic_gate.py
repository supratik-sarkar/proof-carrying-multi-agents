"""PCG-MAS v3.4 Vector Semantic Gate.

Implements Requirement C:
- Multi-dimensional semantic certification; NO support x coverage product score
- Obligation coverage C_obl >= tau_cov
- Support lower-quantile / critical margin rule Q_alpha(s_i) >= tau_E
- Contradiction ceiling max_i pC_i <= tau_C
- Separate evaluation of critical vs non-critical obligations
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from pcg.v3_4.states import RawVerifierState
from pcg.v3_4.obligations import TaskObligation, compute_obligation_coverage, verify_critical_obligations

@dataclass
class NLIWindowScore:
    window_idx: int
    window_text: str
    pe: float
    pc: float
    pn: float
    margin: float

@dataclass
class SemanticVerificationResult:
    raw_state: RawVerifierState
    obligation_coverage: float
    min_margin: float
    max_contradiction: float
    certified_obligations: List[str]
    uncertified_obligations: List[str]
    critical_obligations_satisfied: bool
    evidence_units_evaluated: int
    gate_details: Dict[str, Any]

def evaluate_vector_semantic_gate(
    obligations: List[TaskObligation],
    obligation_scores: Dict[str, List[NLIWindowScore]], # obl_id -> list of window scores
    tau_cov: float = 1.0,
    tau_e: float = 0.05,
    tau_c: float = 0.30,
    support_quantile: float = 0.0 # 0.0 means min margin across obligations
) -> SemanticVerificationResult:
    """Evaluates the multi-dimensional vector semantic gate.
    
    CRITICAL: Does NOT use a scalar product such as support * coverage.
    Uses explicit conjunctive conditions over (coverage, margin, contradiction).
    """
    if not obligations:
        return SemanticVerificationResult(
            raw_state=RawVerifierState.NOT_APPLICABLE,
            obligation_coverage=1.0,
            min_margin=1.0,
            max_contradiction=0.0,
            certified_obligations=[],
            uncertified_obligations=[],
            critical_obligations_satisfied=True,
            evidence_units_evaluated=0,
            gate_details={"reason": "No semantic obligations defined for domain"}
        )
    
    certified_obl_ids = set()
    uncertified_obl_ids = set()
    all_margins = []
    all_pcs = []
    total_evidence_units = 0

    for obl in obligations:
        scores = obligation_scores.get(obl.obligation_id, [])
        total_evidence_units += len(scores)
        if not scores:
            uncertified_obl_ids.add(obl.obligation_id)
            continue
            
        # For this obligation, select the best candidate evidence window (max margin)
        best_window = max(scores, key=lambda w: w.margin)
        all_margins.append(best_window.margin)
        all_pcs.append(best_window.pc)
        
        # An obligation is certified if its best window satisfies margin and contradiction
        is_certified = (best_window.margin >= tau_e) and (best_window.pc <= tau_c)
        if is_certified:
            certified_obl_ids.add(obl.obligation_id)
        else:
            uncertified_obl_ids.add(obl.obligation_id)
            
    cov = compute_obligation_coverage(obligations, certified_obl_ids)
    crit_satisfied = verify_critical_obligations(obligations, certified_obl_ids)
    
    min_m = min(all_margins) if all_margins else -1.0
    max_c = max(all_pcs) if all_pcs else 1.0
    
    if all_margins:
        q_m = float(np.percentile(all_margins, support_quantile * 100))
    else:
        q_m = -1.0
        
    # Explicit Vector Gate Conjunction:
    # 1. Coverage >= tau_cov
    # 2. Critical obligations all satisfied
    # 3. Support quantile >= tau_e
    # 4. Contradiction ceiling <= tau_c
    cov_pass = cov >= (tau_cov - 1e-6)
    supp_pass = q_m >= tau_e
    contra_pass = max_c <= tau_c
    
    if cov_pass and crit_satisfied and supp_pass and contra_pass:
        state = RawVerifierState.PASS
    else:
        state = RawVerifierState.FAIL
        
    details = {
        "tau_cov": tau_cov,
        "tau_e": tau_e,
        "tau_c": tau_c,
        "coverage_pass": cov_pass,
        "critical_satisfied": crit_satisfied,
        "support_pass": supp_pass,
        "contradiction_pass": contra_pass,
        "quantile_margin": q_m
    }
    
    return SemanticVerificationResult(
        raw_state=state,
        obligation_coverage=cov,
        min_margin=min_m,
        max_contradiction=max_c,
        certified_obligations=list(certified_obl_ids),
        uncertified_obligations=list(uncertified_obl_ids),
        critical_obligations_satisfied=crit_satisfied,
        evidence_units_evaluated=total_evidence_units,
        gate_details=details
    )
