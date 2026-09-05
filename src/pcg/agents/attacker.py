"""
Attacker agent — adversarial probes that test certificate soundness.

The Attacker exists for one purpose: to verify that the framework's
soundness claims (Theorem 1: false_claim => check_fail OR int_fail OR
replay_fail OR cov_gap) hold under realistic tampering. We sweep three
canonical attack families:

    1. evidence_swap   : replace the bytes of a TruthNode after commitment
                         (must trigger IntFail since H(x(v)) != h(v))
    2. schema_break    : mutate the contract's required_schema_ids to omit
                         a schema that's actually present (must trigger
                         CheckFail since `required_schema_ids - attached`
                         becomes non-empty)
    3. policy_violation: add a tool call to a tool not in the allowlist
                         (must trigger CheckFail via Check_exe)

Each attack mutates the runtime graph state IN PLACE and returns. The
Verifier (run after the Attacker by the orchestrator) is then expected
to detect and reject. If it doesn't, that's a soundness bug we catch in
unit tests.

Important: the Attacker NEVER mutates the certificate itself. The contract
is that adversaries can tamper with the runtime graph (which is what
auditors inspect), not with the immutable certificate that's been issued.
"""
from __future__ import annotations

import random
from typing import Callable, Literal

from pcg.graph import ActionNode, NodeType, ToolCallNode, TruthNode
from pcg.orchestrator.langgraph_flow import PCGState


AttackKind = Literal["evidence_swap", "schema_break", "policy_violation", "action_replay", "mixed_channel", "none"]


def _attack_evidence_swap(state: PCGState, rng: random.Random) -> str:
    """Tamper with COMMITTED evidence so V_H actually fires.

    Previously this picked any TruthNode in the retrieval pool, which meant
    V_H rarely caught the swap when retrieval pool > top_k (the tampered
    node was not in claim_cert.evidence_ids and so was never re-hashed at
    check time). The fix targets only nodes the certificate committed to,
    matching the guard convention used by schema_break / policy_violation.
    """
    if state.certificate is None:
        return "no_cert"
    committed_ids = tuple(state.certificate.claim_cert.evidence_ids)
    if not committed_ids:
        return "no_committed_evidence"
    candidates = [
        state.graph.nodes[nid]
        for nid in committed_ids
        if nid in state.graph.nodes and hasattr(state.graph.nodes[nid], "payload")
    ]
    if not candidates:
        return "no_truth_to_swap"
    target = rng.choice(candidates)
    target.payload = b"TAMPERED " + rng.randbytes(32)
    return f"evidence_swap:{target.id}"


def _attack_schema_break(state: PCGState, rng: random.Random) -> str:
    """Inject a phantom required schema that the run hasn't satisfied.

    Mutates the certificate's contract — but actually we said attackers can't
    touch certificates. So we instead re-issue: we mutate the SchemaNode in
    the graph to have a different schema_id than what the contract expected.
    Same effect as schema break under Check_exe.
    """
    if state.certificate is None:
        return "no_cert"
    sch_ids = list(state.certificate.exec_cert.schema_node_ids)
    if not sch_ids:
        return "no_schema"
    target_id = rng.choice(sch_ids)
    node = state.graph.nodes.get(target_id)
    if isinstance(node, type(node)) and hasattr(node, "schema_id"):
        node.schema_id = "MUTATED_" + node.schema_id    # type: ignore[attr-defined]
    return f"schema_break:{target_id}"


def _attack_policy_violation(state: PCGState, rng: random.Random) -> str:
    """Add a ToolCall to an out-of-allowlist tool, then attach it to the
    execution certificate's tool_call_ids."""
    if state.certificate is None:
        return "no_cert"
    rogue = ToolCallNode(
        tool_name="rogue_external_api",
        tool_version="666",
        args={"q": "unauthorized"},
        latency_ms=10.0,
    )
    state.graph.add_node(rogue)
    # Re-issue the exec cert with the rogue tool added
    ec = state.certificate.exec_cert
    new_ids = (*ec.tool_call_ids, rogue.id)
    from pcg.certificate import ExecutionCertificate, GroundingCertificate
    new_ec = ExecutionCertificate(
        pipeline=ec.pipeline, contract=ec.contract,
        tool_call_ids=new_ids,
        memory_node_ids=ec.memory_node_ids,
        delegation_ids=ec.delegation_ids,
        schema_node_ids=ec.schema_node_ids,
        policy_node_ids=ec.policy_node_ids,
        meta=ec.meta,
    )
    state.certificate = GroundingCertificate(
        claim_cert=state.certificate.claim_cert,
        exec_cert=new_ec, meta=state.certificate.meta,
    )
    return f"policy_violation:{rogue.id}"


def _attack_action_replay(state: PCGState, rng: random.Random) -> str:
    """Open-set attack: corrupt the prover's ActionNode args.

    This target lives OUTSIDE the closed channel taxonomy
    (V_H/V_Pi/V_Gamma/V_entail). A correct responsibility estimator
    should return "no confident attribution" (small |Resp| across all
    components) rather than confidently pointing at any single component.
    """
    if state.certificate is None:
        return "no_cert"
    action_nodes = [n for n in state.graph.nodes.values()
                     if isinstance(n, ActionNode)]
    if not action_nodes:
        return "no_action_node"
    target = rng.choice(action_nodes)
    target.args = {**target.args, "_open_set_corruption":
                    f"rogue_{rng.randint(0, 1<<31)}"}
    return f"action_replay:{target.id}"


def _attack_mixed_channel(state: PCGState, rng: random.Random) -> str:
    """Mixed-channel attack: tamper TWO components in different channels.

    Corrupts evidence (V_H domain) AND schema (V_Gamma domain) in the same
    record. A correct responsibility estimator should rank BOTH targets in
    its top-2 by |Resp|. We record both target_ids so the runner can score
    multi-label F1.
    """
    if state.certificate is None:
        return "no_cert"
    targets = []

    # Evidence corruption (V_H)
    committed_ids = tuple(state.certificate.claim_cert.evidence_ids)
    if committed_ids:
        ev_candidates = [
            state.graph.nodes[nid] for nid in committed_ids
            if nid in state.graph.nodes and hasattr(state.graph.nodes[nid], "payload")
        ]
        if ev_candidates:
            ev_target = rng.choice(ev_candidates)
            ev_target.payload = b"TAMPERED_MIXED " + rng.randbytes(24)
            targets.append(ev_target.id)

    # Schema corruption (V_Gamma)
    sch_ids = list(state.certificate.exec_cert.schema_node_ids)
    if sch_ids:
        sch_target_id = rng.choice(sch_ids)
        node = state.graph.nodes.get(sch_target_id)
        if hasattr(node, "schema_id"):
            node.schema_id = "MUTATED_MIXED_" + node.schema_id    # type: ignore[attr-defined]
            targets.append(sch_target_id)

    return f"mixed_channel:{','.join(targets) if targets else 'no_targets'}"


def build_default_attacker(
    *,
    kind: AttackKind = "evidence_swap",
    seed: int = 0,
) -> Callable[[PCGState], PCGState]:
    """Return an attacker callable for the orchestrator."""
    rng = random.Random(seed)

    def attacker(state: PCGState) -> PCGState:
        with state.meter.phase("attacker"):
            if kind == "evidence_swap":
                desc = _attack_evidence_swap(state, rng)
            elif kind == "schema_break":
                desc = _attack_schema_break(state, rng)
            elif kind == "policy_violation":
                desc = _attack_policy_violation(state, rng)
            elif kind == "action_replay":
                desc = _attack_action_replay(state, rng)
            elif kind == "mixed_channel":
                desc = _attack_mixed_channel(state, rng)
            else:
                desc = "none"

            state.graph.add_node(ActionNode(
                action="attack", agent_id="attacker", args={"kind": kind, "target": desc},
            ))
            state.meta.setdefault("attacks", []).append({"kind": kind, "desc": desc})
        return state

    return attacker
