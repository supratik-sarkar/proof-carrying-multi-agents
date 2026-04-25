"""
Prover agent.

Responsibilities (Appendix D.4):
    1. Retrieve a candidate evidence set S from the example's evidence pool
    2. Run the LLM to produce a candidate claim c with confidence p
    3. Construct the replayable pipeline Pi (BM25 -> concat -> answer-extract)
    4. Construct the execution contract Gamma (tools allowed, schemas, policies)
    5. Hash all evidence and assemble the unified GroundingCertificate Z

Every measurable operation is wrapped in `state.meter.phase(...)` so R5
(token/time overhead) is a direct measurement.

Determinism: when `temperature=0` and seed is fixed, the same example produces
bit-identical certificate twice (modulo the LLM's own determinism — which is
enforced by HFLocalBackend with manual_seed and is best-effort on remote API).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from pcg.backends.base import LLMBackend
from pcg.certificate import (
    ClaimCertificate,
    ExecutionCertificate,
    ExecutionContract,
    GroundingCertificate,
    ReplayableStep,
)
from pcg.commitments import H
from pcg.datasets.base import EvidenceItem, QAExample
from pcg.graph import (
    AgenticRuntimeGraph,
    ActionNode,
    ClaimNode,
    EdgeType,
    MessageNode,
    PolicyNode,
    SchemaNode,
    SourceNode,
    ToolCallNode,
    TruthNode,
)
from pcg.orchestrator.langgraph_flow import PCGState
from pcg.retrieval import BM25Index


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


_QA_PROMPT = """\
You are a careful assistant. Use ONLY the provided context to answer.
If the context doesn't contain the answer, say "I don't know."
Give a brief answer (1-5 words when possible).

Context:
{context}

Question: {question}

Answer:"""


_TOOL_PROMPT = """\
You are an assistant that uses tools. The tool's output is shown below.
Extract the final numeric or string answer.

Tool output:
{tool_output}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------


def _estimate_confidence(
    answer_text: str,
    retrieval_scores: list[float],
    backend_logprobs: list[float] | None,
) -> float:
    """Heuristic confidence in [0, 1].

    Without per-token log-probs, we use a simple combination:
        conf = sigma( w1 * top_retrieval_score_z + w2 * has_substantive_answer )

    With log-probs, we average the per-token log-probs of the answer span and
    map through a sigmoid. The Calibrator (pcg.risk.Calibrator) is responsible
    for turning these raw scores into calibrated probabilities later — the
    Prover only needs to provide a monotone signal.
    """
    import math
    has_answer = bool(answer_text.strip()) and answer_text.lower().strip() not in {
        "i don't know.", "i don't know", "unknown", ""
    }
    base = 0.85 if has_answer else 0.05

    # Retrieval contribution: clip top score, normalize. BM25 scores live
    # roughly in [0, 30] for small pools; we map via tanh.
    top_score = retrieval_scores[0] if retrieval_scores else 0.0
    retr_signal = math.tanh(top_score / 5.0)   # in (-1, 1)

    # Logprob contribution if available
    lp_signal = 0.0
    if backend_logprobs:
        avg_lp = sum(backend_logprobs) / len(backend_logprobs)
        # avg_lp is negative. Map to [0, 1] via clipped exponential.
        lp_signal = max(0.0, min(1.0, math.exp(avg_lp)))

    raw = 0.55 * base + 0.30 * (0.5 + 0.5 * retr_signal) + 0.15 * lp_signal
    return max(0.01, min(0.99, raw))


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------


def _add_evidence_to_graph(
    graph: AgenticRuntimeGraph,
    evidence: list[EvidenceItem],
) -> tuple[list[str], list[str]]:
    """Add SourceNode + TruthNode pairs for each evidence item.

    Returns (truth_node_ids, evidence_digests) parallel to `evidence`.
    """
    truth_ids: list[str] = []
    digests: list[str] = []
    for ev in evidence:
        # Source node
        src = SourceNode(
            url=ev.source_url,
            authority_id=ev.publisher,           # publisher serves as authority for our datasets
            publisher_id=ev.publisher,
            domain=ev.domain,
        )
        graph.add_node(src)
        # Truth node
        truth = TruthNode(
            payload=ev.text.encode("utf-8"),
            mime="text/plain",
            source_id=src.id,
            attr={"title": ev.title, "is_gold": ev.is_gold, "ext_id": ev.id},
        )
        graph.add_node(truth)
        graph.add_edge(src.id, truth.id, EdgeType.RETRIEVED_FROM)
        truth_ids.append(truth.id)
        digests.append(H(truth.content_for_hash()))
    return truth_ids, digests


def _add_pipeline_to_graph(
    graph: AgenticRuntimeGraph,
    truth_ids: list[str],
    selected_indices: list[int],
    answer_text: str,
    claim_text: str,
) -> tuple[list[str], str, list[ReplayableStep]]:
    """Add ToolCall nodes (representing retrieval + extraction) and edges
    connecting them to a claim node. Returns (tool_call_ids, claim_node_id,
    pipeline_steps).
    """
    # Tool call: BM25 retrieval (deterministic, replayable from the same evidence pool)
    retr_call = ToolCallNode(
        tool_name="bm25_retrieve",
        tool_version="0.2",
        args={"top_k": len(selected_indices), "k1": 1.5, "b": 0.75},
        latency_ms=0.0,
    )
    graph.add_node(retr_call)
    # Connect retrieval to selected truths
    for i in selected_indices:
        graph.add_edge(retr_call.id, truth_ids[i], EdgeType.PRODUCED_BY_TOOL)

    # Claim node
    claim = ClaimNode(
        raw=answer_text,
        canonical=re.sub(r"\s+", " ", answer_text).strip().lower(),
        claim_type="text",
    )
    graph.add_node(claim)
    # Connect every selected truth to the claim through SUPPORTS edges
    for i in selected_indices:
        graph.add_edge(truth_ids[i], claim.id, EdgeType.SUPPORTS)

    # Pipeline: retrieval -> concat -> identity (paper Pi)
    selected_truth_ids = [truth_ids[i] for i in selected_indices]
    pipeline = [
        ReplayableStep(
            op="concat",
            version="0.1",
            params={"delim": "\n"},
            input_ids=tuple(selected_truth_ids),
            output_digest=None,    # filled by Prover after replay
        ),
    ]
    return [retr_call.id], claim.id, pipeline


def _add_execution_contract(
    graph: AgenticRuntimeGraph,
    *,
    task_type: str,
    tools_used: list[str],
) -> tuple[ExecutionContract, list[str], list[str]]:
    """Build the execution contract Gamma and add the corresponding Schema/Policy
    nodes to the graph. Returns (contract, schema_node_ids, policy_node_ids).
    """
    schema = SchemaNode(
        schema_id="qa_answer_v1",
        schema_version="1",
        schema_dict={"type": "object", "required": ["answer"], "properties": {
            "answer": {"type": "string", "maxLength": 200},
        }},
    )
    graph.add_node(schema)
    policy = PolicyNode(
        policy_id="grounded_answer_v1",
        clause_id="must_cite_evidence",
        kind="action_constraint",
        content="Every answer must be supported by at least one cited evidence node.",
    )
    graph.add_node(policy)

    contract = ExecutionContract(
        tool_allowlist=frozenset(tools_used),
        memory_access=frozenset({"short_term:read", "short_term:write"}),
        allowed_delegations=frozenset(),  # Prover acts alone in default config
        required_schema_ids=frozenset({schema.schema_id}),
        required_policy_ids=frozenset({policy.policy_id}),
        max_tool_calls=8,
        max_tokens=2048,
        max_latency_ms=60_000.0,
    )
    return contract, [schema.id], [policy.id]


# ---------------------------------------------------------------------------
# Main Prover function
# ---------------------------------------------------------------------------


@dataclass
class ProverConfig:
    """Tunables exposed to the experiment scripts."""

    top_k: int = 4
    max_answer_tokens: int = 64
    temperature: float = 0.0
    seed: int = 0
    retriever: str = "bm25"            # "bm25" | "dense" | "hybrid"
    use_concat_replay: bool = True


def build_default_prover(
    *,
    backend: LLMBackend,
    config: ProverConfig | None = None,
) -> Callable[[PCGState], PCGState]:
    """Returns a prover callable for the orchestrator.

    The prover is closed over `backend` and `config` so the orchestrator can
    just hand it a state.
    """
    cfg = config or ProverConfig()

    def prover(state: PCGState) -> PCGState:
        ex = state.example

        # 0. Add example-level Action node (so the runtime graph has provenance
        #    for "what the prover did").
        with state.meter.phase("prover_setup"):
            action_id = state.graph.add_node(ActionNode(
                action="answer", agent_id="prover",
                args={"task_type": ex.task_type, "retriever": cfg.retriever},
            ))

        # 1. Add evidence to graph and hash it
        with state.meter.phase("prover_commit_evidence"):
            ev_list = list(ex.evidence)
            truth_ids, digests = _add_evidence_to_graph(state.graph, ev_list)
            # One hash op per evidence item — counted in the meter
            for _ in truth_ids:
                state.meter.record_hash()

        # 2. Retrieval
        with state.meter.phase("prover_retrieval"):
            idx = BM25Index.build(ev_list)
            hits = idx.search(ex.question, top_k=cfg.top_k)
            # Map hits back to indices in ev_list
            id_to_pos = {ev.id: i for i, ev in enumerate(ev_list)}
            selected_pos = [id_to_pos[item.id] for item, _ in hits]
            retrieval_scores = [s for _, s in hits]

        # 3. Build the prompt and call the LLM
        with state.meter.phase("prover_llm_gen"):
            if ex.task_type == "tool_use":
                # For tool-use examples, bypass retrieval and use the tool log directly
                tool_evidence = [ev for ev in ev_list if ev.publisher == "tool"]
                tool_output = "\n".join(ev.text for ev in tool_evidence) if tool_evidence else ""
                prompt = _TOOL_PROMPT.format(question=ex.question, tool_output=tool_output)
            else:
                ctx = "\n\n".join(f"[{i+1}] {hits[i][0].title}: {hits[i][0].text}"
                                  for i in range(len(hits)))
                prompt = _QA_PROMPT.format(context=ctx, question=ex.question)
            gen_out = backend.generate(
                prompt,
                max_tokens=cfg.max_answer_tokens,
                temperature=cfg.temperature,
                seed=cfg.seed,
                stop=["\n\n", "Question:", "Context:"],
            )
            state.meter.record_tokens(tokens_in=gen_out.tokens_in, tokens_out=gen_out.tokens_out)
            answer_text = gen_out.text.strip()

        # 4. Add a MessageNode for the LLM call (auditability)
        msg = MessageNode(
            from_agent="prover", to_agent="env",
            role="assistant", content=answer_text,
            n_tokens=gen_out.tokens_out,
        )
        state.graph.add_node(msg)

        # 5. Build pipeline + claim node
        with state.meter.phase("prover_pipeline"):
            tool_call_ids, claim_id, pipeline = _add_pipeline_to_graph(
                state.graph,
                truth_ids=truth_ids,
                selected_indices=selected_pos,
                answer_text=answer_text,
                claim_text=ex.question,
            )

        # 6. Replay locally to compute the y digest the certificate will commit to
        from pcg.checker import build_default_replayer
        replayer = build_default_replayer()
        # Selected truths in order, concatenated
        selected_truth_ids = [truth_ids[i] for i in selected_pos]
        replay_step = ReplayableStep(
            op="concat", version="0.1", params={"delim": "\n"},
            input_ids=tuple(selected_truth_ids), output_digest=None,
        )
        y_bytes = replayer.run(replay_step, state.graph)
        y_digest = H(y_bytes)
        # Update pipeline with the digest
        pipeline = [
            ReplayableStep(
                op=p.op, version=p.version, params=p.params,
                input_ids=p.input_ids, output_digest=y_digest,
            ) for p in pipeline
        ]

        # 7. Build execution contract
        with state.meter.phase("prover_contract"):
            tools_used: list[str] = ["bm25_retrieve"]
            if ex.task_type == "tool_use":
                tools_used.append("calculator")
            contract, schema_ids, policy_ids = _add_execution_contract(
                state.graph, task_type=ex.task_type, tools_used=tools_used,
            )

        # 8. Estimate confidence and build the certificate
        confidence = _estimate_confidence(
            answer_text, retrieval_scores, gen_out.logprobs,
        )

        claim_cert = ClaimCertificate(
            claim_id=claim_id,
            evidence_ids=tuple(selected_truth_ids),
            evidence_digests=tuple(digests[i] for i in selected_pos),
            pipeline=tuple(pipeline),
            confidence=confidence,
            replay_output_digest=y_digest,
            meta={"backend": gen_out.backend, "retriever": cfg.retriever,
                  "seed": cfg.seed, "raw_answer": answer_text},
        )
        exec_cert = ExecutionCertificate(
            pipeline=tuple(pipeline),
            contract=contract,
            tool_call_ids=tuple(tool_call_ids),
            memory_node_ids=(),
            delegation_ids=(),
            schema_node_ids=tuple(schema_ids),
            policy_node_ids=tuple(policy_ids),
            meta={"agent": "prover"},
        )
        state.certificate = GroundingCertificate(
            claim_cert=claim_cert, exec_cert=exec_cert,
            meta={"task_type": ex.task_type, "example_id": ex.id},
        )
        state.meta["raw_answer"] = answer_text
        state.meta["retrieval_scores"] = retrieval_scores
        return state

    return prover
