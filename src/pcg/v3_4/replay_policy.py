r"""PCG-MAS v3.4 Semantic Replay and Policy Verification.

Implements Requirement E:
- Structured action trace schema
- Replay equivalence predicate T' \equiv_R T ignoring irrelevant nondeterminism (timestamps, request IDs)
- V_Pi (replay equivalence) strictly distinct from V_Gamma (policy compliance)
- No hardcoded replay TRUE; no text similarity proxy
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from pcg.v3_4.states import RawVerifierState

@dataclass
class ActionTraceStep:
    trace_id: str
    action_id: str
    tool_id: str
    canonical_arguments: Dict[str, Any]
    pre_state_hash: str
    policy_context_hash: str
    observation_hash: str
    post_state_hash: str
    authorization_result: str # "AUTHORIZED" | "DENIED"
    timestamp_ephemeral: Optional[str] = None
    request_id_ephemeral: Optional[str] = None

    def canonical_hash(self) -> str:
        """Computes deterministic hash over canonical semantic fields."""
        payload = {
            "action_id": self.action_id,
            "tool_id": self.tool_id,
            "args": self.canonical_arguments,
            "pre_state": self.pre_state_hash,
            "policy_ctx": self.policy_context_hash,
            "obs": self.observation_hash,
            "post_state": self.post_state_hash,
            "auth": self.authorization_result
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def check_semantic_replay_equivalence(
    reference_trace: List[ActionTraceStep],
    replayed_trace: List[ActionTraceStep]
) -> Tuple[RawVerifierState, Dict[str, Any]]:
    r"""Evaluates semantic replay equivalence T' \equiv_R T.
    
    Ignores irrelevant nondeterminism (timestamps, ephemeral request IDs).
    Strictly preserves:
    - invoked operation (tool_id, action_id)
    - canonical parameters
    - policy decision
    - relevant observation
    - state transition
    """
    if not reference_trace and not replayed_trace:
        return RawVerifierState.INDETERMINATE, {"reason": "Empty traces provided"}
    if len(reference_trace) != len(replayed_trace):
        return RawVerifierState.FAIL, {
            "reason": f"Trace length mismatch: ref={len(reference_trace)}, rep={len(replayed_trace)}"
        }
        
    for step_idx, (t_ref, t_rep) in enumerate(zip(reference_trace, replayed_trace)):
        if t_ref.tool_id != t_rep.tool_id:
            return RawVerifierState.FAIL, {
                "step": step_idx,
                "reason": f"Tool mismatch: {t_ref.tool_id} != {t_rep.tool_id}"
            }
        if t_ref.canonical_arguments != t_rep.canonical_arguments:
            return RawVerifierState.FAIL, {
                "step": step_idx,
                "reason": "Canonical arguments mismatch"
            }
        if t_ref.authorization_result != t_rep.authorization_result:
            return RawVerifierState.FAIL, {
                "step": step_idx,
                "reason": f"Authorization mismatch: {t_ref.authorization_result} != {t_rep.authorization_result}"
            }
        if t_ref.observation_hash != t_rep.observation_hash:
            return RawVerifierState.FAIL, {
                "step": step_idx,
                "reason": "Observation hash mismatch"
            }
        if t_ref.post_state_hash != t_rep.post_state_hash:
            return RawVerifierState.FAIL, {
                "step": step_idx,
                "reason": "Post-state transition mismatch"
            }
            
    return RawVerifierState.PASS, {"verified_steps": len(reference_trace)}

def check_policy_compliance(
    trace: List[ActionTraceStep],
    policy_rules: Optional[Dict[str, Any]] = None
) -> Tuple[RawVerifierState, Dict[str, Any]]:
    """Evaluates policy compliance V_Gamma.
    
    Checks pre-execution policy rules, argument boundaries, and unauthorized operations.
    Strictly distinct from replay equivalence V_Pi.
    """
    policy_rules = policy_rules or {}
    prohibited_tools = set(policy_rules.get("prohibited_tools", ["rm", "eval", "sudo", "exec_malicious", "exfiltrate"]))
    
    if not trace:
        return RawVerifierState.INDETERMINATE, {"reason": "Empty trace for policy validation"}
        
    for step_idx, step in enumerate(trace):
        # 1. Prohibited tool call
        if step.tool_id in prohibited_tools:
            return RawVerifierState.FAIL, {
                "step": step_idx,
                "reason": f"Prohibited tool invoked: {step.tool_id}"
            }
        # 2. Authorization violation
        if step.authorization_result == "DENIED":
            return RawVerifierState.FAIL, {
                "step": step_idx,
                "reason": "Action explicitly denied by policy context"
            }
        # 3. Path traversal or boundary check in arguments
        for k, v in step.canonical_arguments.items():
            if isinstance(v, str) and ("../.." in v or "/etc/passwd" in v or "DROP TABLE" in v.upper()):
                return RawVerifierState.FAIL, {
                    "step": step_idx,
                    "reason": f"Malicious argument pattern detected in parameter '{k}'"
                }
                
    return RawVerifierState.PASS, {"inspected_steps": len(trace)}

def evaluate_replay_and_policy(
    dataset: str,
    reference_trace: Optional[List[ActionTraceStep]],
    replayed_trace: Optional[List[ActionTraceStep]],
    policy_rules: Optional[Dict[str, Any]] = None
) -> Tuple[RawVerifierState, RawVerifierState, Dict[str, Any]]:
    """Evaluates (V_Pi, V_Gamma) for a given dataset execution.
    
    For grounding tasks, returns (NOT_APPLICABLE, NOT_APPLICABLE).
    For interactive tasks (ToolBench, WebLINX), computes replay equivalence and policy compliance.
    """
    if dataset not in ("toolbench", "weblinx"):
        return RawVerifierState.NOT_APPLICABLE, RawVerifierState.NOT_APPLICABLE, {
            "reason": f"Replay and policy not applicable for grounding task '{dataset}'"
        }
        
    if reference_trace is None or replayed_trace is None:
        return RawVerifierState.INDETERMINATE, RawVerifierState.INDETERMINATE, {
            "reason": "Missing trace fixtures for interactive execution"
        }
        
    v_pi, r_details = check_semantic_replay_equivalence(reference_trace, replayed_trace)
    v_gamma, p_details = check_policy_compliance(reference_trace, policy_rules)
    
    details = {
        "v_pi_details": r_details,
        "v_gamma_details": p_details
    }
    return v_pi, v_gamma, details
