"""PCG-MAS v3.4 Verifier States and Operational Acceptance.

Implements Requirement D:
- Raw states: PASS, FAIL, INDETERMINATE, NOT_APPLICABLE, NOT_EVALUATED
- Explicit separation between raw state and effective binary operational bit
- NOT_APPLICABLE is NEVER stored as raw PASS
- NOT_EVALUATED is NEVER stored as raw FAIL
- Strict fail-closed semantics for INDETERMINATE and NOT_EVALUATED
"""
from enum import Enum
from typing import Dict, Any, Optional

class RawVerifierState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_EVALUATED = "NOT_EVALUATED"

def effective_binary_bit(state: RawVerifierState) -> int:
    """Computes operational conjunction contribution for a single channel.
    
    - PASS -> 1
    - FAIL -> 0
    - INDETERMINATE -> 0 (fail-closed)
    - NOT_EVALUATED -> 0 (short-circuit or unreached)
    - NOT_APPLICABLE -> 1 (vacuous satisfaction in conjunction)
    """
    if state == RawVerifierState.PASS:
        return 1
    elif state == RawVerifierState.NOT_APPLICABLE:
        return 1
    elif state in (RawVerifierState.FAIL, RawVerifierState.INDETERMINATE, RawVerifierState.NOT_EVALUATED):
        return 0
    raise ValueError(f"Unknown verifier state: {state}")

def compute_operational_acceptance(channel_states: Dict[str, RawVerifierState]) -> int:
    """Computes joint operational acceptance across all channels:
    
    Check = 1 iff for all channels k, effective_binary_bit(k) == 1.
    """
    if not channel_states:
        return 0
    for ch, st in channel_states.items():
        if effective_binary_bit(st) == 0:
            return 0
    return 1
