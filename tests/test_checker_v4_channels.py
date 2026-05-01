from pcg.certificate import (
    ClaimCertificate,
    ExecutionCertificate,
    ExecutionContract,
    GroundingCertificate,
    ReplayableStep,
)
from pcg.checker import Checker, ExactMatchEntailment, build_default_replayer
from pcg.commitments import H
from pcg.graph import AgenticRuntimeGraph, ClaimNode, ToolCallNode, TruthNode


def _base_graph(claim_text: str = "Paris"):
    graph = AgenticRuntimeGraph(run_id="test-run")

    truth = TruthNode(
        id="truth1",
        payload=b"The answer is Paris.",
        mime="text/plain",
    )
    claim = ClaimNode(
        id="claim1",
        raw=claim_text,
        canonical=claim_text,
    )

    graph.add_node(truth)
    graph.add_node(claim)

    return graph, truth, claim


def _cert(
    graph,
    truth,
    *,
    evidence_digest=None,
    replay_digest=None,
    contract=None,
    tool_call_ids=(),
):
    step = ReplayableStep(
        op="identity",
        version="test-v1",
        params={},
        input_ids=(truth.id,),
        output_digest=None,
    )

    cc = ClaimCertificate(
        claim_id="claim1",
        evidence_ids=(truth.id,),
        evidence_digests=(evidence_digest or H(truth.content_for_hash()),),
        pipeline=(step,),
        confidence=0.9,
        replay_output_digest=replay_digest or H(truth.payload),
        meta={"seed": 0},
    )

    ec = ExecutionCertificate(
        pipeline=(step,),
        contract=contract or ExecutionContract(),
        tool_call_ids=tuple(tool_call_ids),
        memory_node_ids=(),
        delegation_ids=(),
        schema_node_ids=(),
        policy_node_ids=(),
        meta={"seed": 0},
    )

    return GroundingCertificate(claim_cert=cc, exec_cert=ec, meta={})


def _checker():
    return Checker(
        entailment=ExactMatchEntailment(case_insensitive=True),
        replayer=build_default_replayer(),
    )


def test_checker_passes_all_four_channels():
    graph, truth, _ = _base_graph("Paris")
    cert = _cert(graph, truth)

    result = _checker().check(cert, graph)

    assert result.passed
    assert result.integrity_ok
    assert result.replay_ok
    assert result.execution_ok
    assert result.entailment_ok
    assert result.V_H and result.V_Pi and result.V_Gamma and result.V_entail


def test_integrity_channel_fails_on_bad_hash():
    graph, truth, _ = _base_graph("Paris")
    cert = _cert(graph, truth, evidence_digest="bad-digest")

    result = _checker().check(cert, graph)

    assert not result.passed
    assert not result.integrity_ok
    assert "hash_mismatch:truth1" in result.reasons


def test_replay_channel_fails_on_bad_replay_digest():
    graph, truth, _ = _base_graph("Paris")
    cert = _cert(graph, truth, replay_digest="bad-replay-digest")

    result = _checker().check(cert, graph)

    assert not result.passed
    assert result.integrity_ok
    assert not result.replay_ok
    assert any(reason.startswith("replay_digest_mismatch") for reason in result.reasons)


def test_entailment_channel_fails_when_claim_not_supported():
    graph, truth, _ = _base_graph("London")
    cert = _cert(graph, truth)

    result = _checker().check(cert, graph)

    assert not result.passed
    assert result.integrity_ok
    assert result.replay_ok
    assert not result.entailment_ok
    assert any(reason.startswith("entailment_rejected") for reason in result.reasons)


def test_execution_channel_fails_on_blocked_tool():
    graph, truth, _ = _base_graph("Paris")

    tool = ToolCallNode(
        id="tool1",
        tool_name="blocked_tool",
        tool_version="v1",
    )
    graph.add_node(tool)

    contract = ExecutionContract(
        tool_allowlist=frozenset({"allowed_tool"}),
    )

    cert = _cert(
        graph,
        truth,
        contract=contract,
        tool_call_ids=("tool1",),
    )

    result = _checker().check(cert, graph)

    assert not result.passed
    assert result.integrity_ok
    assert result.replay_ok
    assert result.entailment_ok
    assert not result.execution_ok
    assert "tool_not_allowed:blocked_tool" in result.reasons