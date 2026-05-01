"""
Property tests for the Check predicate.

Invariants we verify:
    1. Determinism: Check(Z, G) == Check(Z, G) (called twice).
    2. Hash-tampering soundness: mutating any TruthNode payload causes
       Check.integrity_ok to flip to False.
    3. Replay digest soundness: changing the replay output digest in the
       certificate causes Check.replay_ok to flip to False.
    4. Round-trip: GroundingCertificate.to_json() / from_json() preserves
       canonical bytes (digest equality).
    5. Merkle prefix verification: AuditLog.verify_prefix is correct under
       extension and detects re-ordering.
"""
from __future__ import annotations

import json

import pytest

from pcg.certificate import (
    ClaimCertificate,
    ExecutionCertificate,
    ExecutionContract,
    GroundingCertificate,
    ReplayableStep,
)
from pcg.checker import (
    Checker,
    ExactMatchEntailment,
    build_default_replayer,
)
from pcg.commitments import H, AuditLog
from pcg.graph import (
    AgenticRuntimeGraph,
    ClaimNode,
    EdgeType,
    SchemaNode,
    PolicyNode,
    SourceNode,
    ToolCallNode,
    TruthNode,
)


# ---------------------------------------------------------------------------
# Fixture: a tiny graph + valid certificate
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_graph_and_cert():
    """A graph with one source, one truth, one tool call, one schema, one
    policy, one claim — and a valid GroundingCertificate over it."""
    g = AgenticRuntimeGraph()

    src = SourceNode(url="https://wiki.test/Paris", authority_id="wiki",
                     publisher_id="wiki", domain="wiki.test")
    g.add_node(src)

    truth = TruthNode(
        payload=b"Paris is the capital of France.",
        mime="text/plain",
        source_id=src.id,
        attr={"title": "Paris", "is_gold": True},
    )
    g.add_node(truth)
    g.add_edge(src.id, truth.id, EdgeType.RETRIEVED_FROM)

    tool = ToolCallNode(
        tool_name="bm25_retrieve",
        tool_version="0.2",
        args={"top_k": 1, "k1": 1.5, "b": 0.75},
        latency_ms=0.0,
    )
    g.add_node(tool)
    g.add_edge(tool.id, truth.id, EdgeType.PRODUCED_BY_TOOL)

    schema = SchemaNode(schema_id="qa_v1", schema_version="1", schema_dict={})
    g.add_node(schema)
    policy = PolicyNode(policy_id="ground_v1", clause_id="must_cite",
                        kind="action_constraint", content="must cite")
    g.add_node(policy)

    claim = ClaimNode(raw="Paris", canonical="paris", claim_type="text")
    g.add_node(claim)
    g.add_edge(truth.id, claim.id, EdgeType.SUPPORTS)

    # Replay: concat of the truth payload
    replayer = build_default_replayer()
    step = ReplayableStep(
        op="concat", version="0.1", params={"delim": "\n"},
        input_ids=(truth.id,),
    )
    y = replayer.run(step, g)
    y_digest = H(y)

    pipeline = (ReplayableStep(
        op="concat", version="0.1", params={"delim": "\n"},
        input_ids=(truth.id,), output_digest=y_digest,
    ),)

    contract = ExecutionContract(
        tool_allowlist=frozenset({"bm25_retrieve"}),
        memory_access=frozenset(),
        allowed_delegations=frozenset(),
        required_schema_ids=frozenset({"qa_v1"}),
        required_policy_ids=frozenset({"ground_v1"}),
    )

    cc = ClaimCertificate(
        claim_id=claim.id,
        evidence_ids=(truth.id,),
        evidence_digests=(H(truth.content_for_hash()),),
        pipeline=pipeline,
        confidence=0.9,
        replay_output_digest=y_digest,
    )
    ec = ExecutionCertificate(
        pipeline=pipeline, contract=contract,
        tool_call_ids=(tool.id,),
        memory_node_ids=(), delegation_ids=(),
        schema_node_ids=(schema.id,), policy_node_ids=(policy.id,),
    )
    cert = GroundingCertificate(claim_cert=cc, exec_cert=ec)

    checker = Checker(
        entailment=ExactMatchEntailment(case_insensitive=True),
        replayer=replayer,
    )
    return g, cert, checker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_check_passes_on_valid(fixture_graph_and_cert):
    g, cert, checker = fixture_graph_and_cert
    r = checker.check(cert, g)
    assert r.passed, f"Expected passed; reasons: {r.reasons}"
    assert r.integrity_ok and r.replay_ok and r.entailment_ok and r.execution_ok


def test_check_is_deterministic(fixture_graph_and_cert):
    g, cert, checker = fixture_graph_and_cert
    r1 = checker.check(cert, g)
    r2 = checker.check(cert, g)
    assert r1.passed == r2.passed
    assert r1.replay_output_digest == r2.replay_output_digest


def test_evidence_tampering_fails_integrity(fixture_graph_and_cert):
    g, cert, checker = fixture_graph_and_cert
    # Find the truth node and mutate its payload
    truth_id = cert.claim_cert.evidence_ids[0]
    truth_node = g.nodes[truth_id]
    truth_node.payload = b"TAMPERED PAYLOAD"
    r = checker.check(cert, g)
    assert not r.passed
    assert not r.integrity_ok
    assert any("hash_mismatch" in reason for reason in r.reasons)


def test_replay_digest_tampering_fails_replay(fixture_graph_and_cert):
    g, cert, checker = fixture_graph_and_cert
    bad_cc = ClaimCertificate(
        claim_id=cert.claim_cert.claim_id,
        evidence_ids=cert.claim_cert.evidence_ids,
        evidence_digests=cert.claim_cert.evidence_digests,
        pipeline=cert.claim_cert.pipeline,
        confidence=cert.claim_cert.confidence,
        replay_output_digest="0" * 64,    # wrong digest
    )
    bad_cert = GroundingCertificate(claim_cert=bad_cc, exec_cert=cert.exec_cert)
    r = checker.check(bad_cert, g)
    assert not r.passed
    assert not r.replay_ok


def test_unknown_tool_fails_execution(fixture_graph_and_cert):
    g, cert, checker = fixture_graph_and_cert
    # Add a rogue tool node and update the cert's tool_call_ids
    rogue = ToolCallNode(tool_name="not_in_allowlist", tool_version="x", args={})
    g.add_node(rogue)
    bad_ec = ExecutionCertificate(
        pipeline=cert.exec_cert.pipeline,
        contract=cert.exec_cert.contract,
        tool_call_ids=(*cert.exec_cert.tool_call_ids, rogue.id),
        memory_node_ids=cert.exec_cert.memory_node_ids,
        delegation_ids=cert.exec_cert.delegation_ids,
        schema_node_ids=cert.exec_cert.schema_node_ids,
        policy_node_ids=cert.exec_cert.policy_node_ids,
    )
    bad_cert = GroundingCertificate(claim_cert=cert.claim_cert, exec_cert=bad_ec)
    r = checker.check(bad_cert, g)
    assert not r.passed
    assert not r.execution_ok


def test_certificate_round_trip(fixture_graph_and_cert):
    g, cert, _ = fixture_graph_and_cert
    s = cert.to_json()
    cert2 = GroundingCertificate.from_json(s)
    # Digests must match — this is the strongest guarantee of identity
    assert cert.digest() == cert2.digest()


def test_merkle_prefix_verification():
    log_a = AuditLog()
    log_b = AuditLog()
    leaves = [H(f"leaf_{i}".encode()) for i in range(5)]
    for leaf in leaves[:3]:
        log_a.append(leaf)
    for leaf in leaves:
        log_b.append(leaf)
    # log_a is a proper prefix of log_b
    assert log_b.verify_prefix(log_a)
    # log_b is NOT a prefix of log_a
    assert not log_a.verify_prefix(log_b)


def test_merkle_detects_reorder():
    log_a = AuditLog()
    log_b = AuditLog()
    leaves = [H(f"leaf_{i}".encode()) for i in range(3)]
    log_a.append(leaves[0])
    log_a.append(leaves[1])
    # Reorder for log_b
    log_b.append(leaves[1])
    log_b.append(leaves[0])
    # Neither should be a prefix of the other
    assert not log_a.verify_prefix(log_b)
    assert not log_b.verify_prefix(log_a)


# ---------------------------------------------------------------------------
# v4 backward compatibility tests
# ---------------------------------------------------------------------------


def test_v4_execution_contract_with_new_fields():
    """Test creating a v4 contract with new fields (allowed_tools, schemas, etc.)."""
    contract = ExecutionContract(
        allowed_tools=frozenset({"bm25", "dpr"}),
        blocked_tools=frozenset({"dangerous_tool"}),
        schemas={
            "qa_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        },
        memory_policy={"allowed_scopes": ["short_term", "persistent"]},
        delegation_policy={"allowed_delegations": ["retriever", "summarizer"]},
        policy_rules=[
            {"policy_id": "ground_v1", "kind": "guardrail", "clause": "must_cite"},
            {"policy_id": "safety_v1", "kind": "guardrail", "clause": "no_harmful_output"},
        ],
        mcp_endpoints={"mcp_server_1": {"url": "http://localhost:8000"}},
        resource_budget={
            "max_tokens": 2048,
            "max_latency_ms": 30000.0,
            "max_tool_calls": 10,
        },
    )
    
    # Verify fields are set
    assert contract.allowed_tools == frozenset({"bm25", "dpr"})
    assert contract.blocked_tools == frozenset({"dangerous_tool"})
    assert "qa_schema" in contract.schemas
    assert contract.resource_budget["max_tokens"] == 2048


def test_v2_backward_compatibility_factory():
    """Test creating v4 contract from v2 fields using from_v2() factory."""
    contract = ExecutionContract.from_v2(
        tool_allowlist=frozenset({"bm25_retrieve"}),
        memory_access=frozenset({"short_term:read", "persistent:read"}),
        allowed_delegations=frozenset({"agent_1"}),
        required_schema_ids=frozenset({"qa_v1"}),
        required_policy_ids=frozenset({"ground_v1"}),
        max_tokens=2048,
        max_latency_ms=30000.0,
        max_tool_calls=10,
    )
    
    # v4 fields should be populated
    assert contract.allowed_tools == frozenset({"bm25_retrieve"})
    assert contract.memory_policy == {"allowed_scopes": ["persistent:read", "short_term:read"]}
    assert "qa_v1" in contract.schemas
    assert any(r.get("policy_id") == "ground_v1" for r in contract.policy_rules)
    
    # v2 fields should also be preserved
    assert contract.tool_allowlist == frozenset({"bm25_retrieve"})
    assert contract.memory_access == frozenset({"short_term:read", "persistent:read"})


def test_v4_contract_getter_methods():
    """Test helper getter methods that work with both v4 and v2 fields."""
    contract = ExecutionContract(
        allowed_tools=frozenset({"tool_a", "tool_b"}),
        memory_policy={"allowed_scopes": ["scope_x", "scope_y"]},
        delegation_policy={"allowed_delegations": ["agent_1"]},
        schemas={"schema_1": {}, "schema_2": {}},
        policy_rules=[
            {"policy_id": "policy_1"},
            {"policy_id": "policy_2"},
        ],
    )
    
    # Getter methods should return v4 field values
    assert contract.get_allowed_tools() == frozenset({"tool_a", "tool_b"})
    assert contract.get_memory_scopes() == frozenset({"scope_x", "scope_y"})
    assert contract.get_allowed_delegations() == frozenset({"agent_1"})
    assert contract.get_required_schema_ids() == frozenset({"schema_1", "schema_2"})
    assert contract.get_required_policy_ids() == frozenset({"policy_1", "policy_2"})


def test_v4_contract_getter_fallback_to_v2():
    """Test that getters fall back to v2 fields when v4 fields are empty."""
    contract = ExecutionContract(
        tool_allowlist=frozenset({"v2_tool"}),
        memory_access=frozenset({"v2_scope"}),
        allowed_delegations=frozenset({"v2_agent"}),
        required_schema_ids=frozenset({"v2_schema"}),
        required_policy_ids=frozenset({"v2_policy"}),
    )
    
    # Getters should return v2 values since v4 fields are empty
    assert contract.get_allowed_tools() == frozenset({"v2_tool"})
    assert contract.get_memory_scopes() == frozenset({"v2_scope"})
    assert contract.get_allowed_delegations() == frozenset({"v2_agent"})
    assert contract.get_required_schema_ids() == frozenset({"v2_schema"})
    assert contract.get_required_policy_ids() == frozenset({"v2_policy"})


def test_load_v2_certificate_json_upgrades_to_v4(fixture_graph_and_cert):
    """Test loading a v2 certificate (no version field) upgrades gracefully to v4."""
    g, cert, checker = fixture_graph_and_cert
    
    # Export as v4 JSON
    json_str = cert.to_json()
    cert_dict = json.loads(json_str)
    
    # Simulate v2 JSON by removing the version field (v2 certs have no version)
    del cert_dict["exec_cert"]["contract"]["version"]
    v2_json_str = json.dumps(cert_dict)
    
    # Load the v2 JSON
    cert_loaded = GroundingCertificate.from_json(v2_json_str)
    
    # Should load successfully and have v4 fields populated from v2 fields
    assert cert_loaded.exec_cert.contract.allowed_tools
    assert cert_loaded.exec_cert.contract.resource_budget is not None


def test_round_trip_v4_certificate_preserves_digest(fixture_graph_and_cert):
    """Test that round-tripping a v4 certificate through JSON preserves digest."""
    g, cert, _ = fixture_graph_and_cert
    original_digest = cert.digest()
    
    # Serialize and deserialize
    json_str = cert.to_json()
    cert_reloaded = GroundingCertificate.from_json(json_str)
    
    # Digest should be identical (canonical form preserved)
    assert cert_reloaded.digest() == original_digest


def test_v4_canonical_bytes_deterministic():
    """Test that ExecutionContract.canonical_bytes() is deterministic."""
    contract = ExecutionContract(
        allowed_tools=frozenset({"tool_b", "tool_a"}),  # Unordered set
        schemas={"schema_a": {}, "schema_b": {}},      # Unordered dict
        resource_budget={"max_tokens": 2048, "max_latency_ms": 30000.0},
    )
    
    # Call canonical_bytes() multiple times
    bytes1 = contract.canonical_bytes()
    bytes2 = contract.canonical_bytes()
    
    # Must be byte-identical (sorted keys, sorted sets)
    assert bytes1 == bytes2
    
    # Create another contract with same values in different order
    contract2 = ExecutionContract(
        allowed_tools=frozenset({"tool_a", "tool_b"}),  # Different order
        schemas={"schema_b": {}, "schema_a": {}},       # Different order
        resource_budget={"max_latency_ms": 30000.0, "max_tokens": 2048},
    )
    
    # Canonical bytes should still be identical
    assert contract.canonical_bytes() == contract2.canonical_bytes()


def test_v4_contract_in_certificate_json_roundtrip(fixture_graph_and_cert):
    """Test full certificate with v4 contract fields survives JSON round-trip."""
    g, orig_cert, checker = fixture_graph_and_cert
    
    # Enhance the contract with v4 fields
    enhanced_contract = ExecutionContract(
        allowed_tools=frozenset({"bm25_retrieve"}),
        blocked_tools=frozenset({"dangerous"}),
        schemas=orig_cert.exec_cert.contract.required_schema_ids and 
                {sid: {"type": "object"} for sid in orig_cert.exec_cert.contract.required_schema_ids}
                or {},
        memory_policy={"allowed_scopes": list(orig_cert.exec_cert.contract.memory_access)},
        delegation_policy={"allowed_delegations": list(orig_cert.exec_cert.contract.allowed_delegations)},
        policy_rules=[
            {"policy_id": pid, "kind": "guardrail"} 
            for pid in orig_cert.exec_cert.contract.required_policy_ids
        ],
        resource_budget={
            "max_tokens": orig_cert.exec_cert.contract.max_tokens,
            "max_latency_ms": orig_cert.exec_cert.contract.max_latency_ms,
            "max_tool_calls": orig_cert.exec_cert.contract.max_tool_calls,
        },
    )
    
    enhanced_exec_cert = ExecutionCertificate(
        pipeline=orig_cert.exec_cert.pipeline,
        contract=enhanced_contract,
        tool_call_ids=orig_cert.exec_cert.tool_call_ids,
        memory_node_ids=orig_cert.exec_cert.memory_node_ids,
        delegation_ids=orig_cert.exec_cert.delegation_ids,
        schema_node_ids=orig_cert.exec_cert.schema_node_ids,
        policy_node_ids=orig_cert.exec_cert.policy_node_ids,
    )
    
    enhanced_cert = GroundingCertificate(
        claim_cert=orig_cert.claim_cert,
        exec_cert=enhanced_exec_cert,
        meta=orig_cert.meta,
    )
    
    # Serialize and deserialize
    json_str = enhanced_cert.to_json()
    reloaded = GroundingCertificate.from_json(json_str)
    
    # v4 fields should survive the round-trip
    assert reloaded.exec_cert.contract.blocked_tools == frozenset({"dangerous"})
    assert reloaded.exec_cert.contract.schemas
    assert reloaded.exec_cert.contract.memory_policy
    assert reloaded.exec_cert.contract.delegation_policy
    assert reloaded.exec_cert.contract.policy_rules
