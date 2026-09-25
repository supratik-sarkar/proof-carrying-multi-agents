"""PCG-MAS v3.4 Two-Stage Execution Controller.

Implements Requirement G:
- Stage 1: Evaluates cheap/deterministic factors (V_H, V_Pi, V_Gamma)
- Stage 2: Semantic verification (V_vdash) evaluated ONLY if Stage 1 permits continuation
- If short-circuited: V_vdash stored strictly as NOT_EVALUATED
- NOT_EVALUATED is never stored as 0 or FAIL
- Preserves accounting of skipped semantic checks
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Callable

from pcg.v3_4.states import RawVerifierState, effective_binary_bit, compute_operational_acceptance

@dataclass
class TwoStageExecutionResult:
    raw_channel_states: Dict[str, RawVerifierState]
    operational_acceptance: int
    stage1_passed: bool
    semantic_short_circuited: bool
    semantic_forward_passes_saved: int
    diagnostics: Dict[str, Any]

def execute_two_stage_certificate(
    v_h: RawVerifierState,
    v_pi: RawVerifierState,
    v_gamma: RawVerifierState,
    semantic_eval_fn: Callable[[], RawVerifierState],
    potential_semantic_passes: int = 8
) -> TwoStageExecutionResult:
    """Executes the two-stage certificate pipeline.
    
    CRITICAL INVARIANTS:
    - If any applicable Stage 1 channel fails or is indeterminate, semantic verifier is short-circuited.
    - Short-circuited semantic verifier is recorded as RawVerifierState.NOT_EVALUATED.
    - NOT_EVALUATED contributes 0 to operational acceptance (fail-closed) but is NOT an evaluated FAIL in channel stats.
    """
    raw_states = {
        "V_H": v_h,
        "V_Pi": v_pi,
        "V_Gamma": v_gamma
    }
    
    # Check if Stage 1 allows continuation
    # Continuation is permitted iff every applicable Stage 1 factor has effective bit 1
    stage1_ok = (effective_binary_bit(v_h) == 1 and 
                 effective_binary_bit(v_pi) == 1 and 
                 effective_binary_bit(v_gamma) == 1)
                 
    if not stage1_ok:
        # Short-circuit Stage 2
        raw_states["V_vdash"] = RawVerifierState.NOT_EVALUATED
        acc = compute_operational_acceptance(raw_states)
        return TwoStageExecutionResult(
            raw_channel_states=raw_states,
            operational_acceptance=acc,
            stage1_passed=False,
            semantic_short_circuited=True,
            semantic_forward_passes_saved=potential_semantic_passes,
            diagnostics={"short_circuit_reason": "Stage 1 check failed or indeterminate"}
        )
        
    # Stage 1 passed, execute Stage 2
    v_vdash = semantic_eval_fn()
    raw_states["V_vdash"] = v_vdash
    acc = compute_operational_acceptance(raw_states)
    
    return TwoStageExecutionResult(
        raw_channel_states=raw_states,
        operational_acceptance=acc,
        stage1_passed=True,
        semantic_short_circuited=False,
        semantic_forward_passes_saved=0,
        diagnostics={"stage2_executed": True}
    )
