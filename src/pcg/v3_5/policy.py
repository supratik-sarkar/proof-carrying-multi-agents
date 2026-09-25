"""Production policy compliance verifier V_Gamma for PCG-MAS v3.5.

V_Gamma evaluates:
- Schema, policy, and authorization compliance on the recorded trace.
- Must NOT depend on the replay verdict (independent of V_Pi).
- Implements decoupling canaries C_Gamma and C_Pi.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pcg.v3_5.core import VerifierState


class PolicyVerifier:
    """Policy compliance verifier V_Gamma."""

    ALLOWED_TOOLS = {
        "search_documentation",
        "calculate_metrics",
        "lookup_patient_record",
        "run_sql_query",
        "browser_navigate",
        "browser_click",
    }

    def verify(
        self,
        recorded_trace: Optional[List[Dict[str, Any]]],
    ) -> Tuple[VerifierState, Dict[str, Any]]:
        """Verify policy compliance of recorded trace.

        Returns:
            (VerifierState, audit_details)
        """
        audit: Dict[str, Any] = {
            "factor": "V_Gamma",
            "violations": [],
        }

        if recorded_trace is None:
            audit["reason"] = "MISSING_RECORDED_TRACE"
            return VerifierState.INDETERMINATE, audit

        if len(recorded_trace) == 0:
            # Empty trace has no policy violations
            return VerifierState.PASS, audit

        for idx, step in enumerate(recorded_trace):
            # Check authorization result
            auth_res = step.get("authorization_result", "AUTHORIZED")
            if auth_res not in ("AUTHORIZED", "ALLOW", "APPROVED"):
                audit["violations"].append({
                    "step": idx,
                    "reason": f"Unauthorized action: authorization_result='{auth_res}'",
                })
                return VerifierState.FAIL, audit

            # Check tool permission
            tool_id = step.get("tool_id", "")
            if tool_id and tool_id not in self.ALLOWED_TOOLS:
                audit["violations"].append({
                    "step": idx,
                    "reason": f"Disallowed tool access: tool_id='{tool_id}'",
                })
                return VerifierState.FAIL, audit

        audit["status"] = "PASS"
        return VerifierState.PASS, audit


def create_decoupling_canaries() -> Dict[str, Any]:
    """Construct decoupling canaries C_Gamma and C_Pi.

    C_Gamma: V_Pi == PASS, V_Gamma == FAIL
    C_Pi: V_Pi == FAIL, V_Gamma == PASS
    """
    # Base authorized action
    authorized_step_1 = {
        "trace_id": "tr_001",
        "parent_span_id": "span_0",
        "action_id": "act_001",
        "tool_id": "search_documentation",
        "tool_version": "1.0",
        "canonical_arguments": {"query": "protocol guidelines"},
        "pre_state_hash": "pre_hash_aaa",
        "policy_context_hash": "ctx_hash_111",
        "authorization_result": "AUTHORIZED",
        "observation_semantic_hash": "obs_hash_bbb",
        "post_state_hash": "post_hash_ccc",
        "environment_snapshot_id": "env_snap_01",
    }

    # Unauthorized action (violates policy)
    unauthorized_step = {
        "trace_id": "tr_002",
        "parent_span_id": "span_0",
        "action_id": "act_002",
        "tool_id": "search_documentation",
        "tool_version": "1.0",
        "canonical_arguments": {"query": "protocol guidelines"},
        "pre_state_hash": "pre_hash_aaa",
        "policy_context_hash": "ctx_hash_111",
        "authorization_result": "DENIED",  # Policy violation!
        "observation_semantic_hash": "obs_hash_bbb",
        "post_state_hash": "post_hash_ccc",
        "environment_snapshot_id": "env_snap_01",
    }

    # Materially mutated replay step (violates replay equivalence)
    materially_mutated_replay_step = {
        "trace_id": "tr_001",
        "parent_span_id": "span_0",
        "action_id": "act_001_MUTATED",  # Material difference!
        "tool_id": "search_documentation",
        "tool_version": "1.0",
        "canonical_arguments": {"query": "different query entirely"},
        "pre_state_hash": "pre_hash_aaa",
        "policy_context_hash": "ctx_hash_111",
        "authorization_result": "AUTHORIZED",
        "observation_semantic_hash": "obs_hash_different",
        "post_state_hash": "post_hash_different",
        "environment_snapshot_id": "env_snap_01",
    }

    return {
        "c_gamma": {
            "name": "C_Gamma",
            "description": "Trajectory replays identically (V_Pi PASS) but contains policy violation (V_Gamma FAIL)",
            "recorded": [unauthorized_step],
            "replayed": [unauthorized_step],  # Identical replay
            "expected_v_pi": VerifierState.PASS,
            "expected_v_gamma": VerifierState.FAIL,
        },
        "c_pi": {
            "name": "C_Pi",
            "description": "Trajectory is fully authorized (V_Gamma PASS) but replay mutates materially (V_Pi FAIL)",
            "recorded": [authorized_step_1],
            "replayed": [materially_mutated_replay_step],  # Material replay failure
            "expected_v_pi": VerifierState.FAIL,
            "expected_v_gamma": VerifierState.PASS,
        },
    }


def verify_policy_domain() -> Dict[str, Any]:
    """Production verification callable for policy domain."""
    canaries = create_decoupling_canaries()
    return {"domain": "policy", "canaries_available": len(canaries)}

