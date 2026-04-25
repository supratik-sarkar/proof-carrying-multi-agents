"""
Replay handlers for the Checker.

Each pipeline step in a ClaimCertificate names an `op` (e.g., "bm25_retrieve",
"span_extract", "nli_filter"). The Checker dispatches replay to the handler
registered here for that op name. A handler is a deterministic function:

    handler(step: ReplayableStep, graph: GraphLike) -> bytes

Handlers MUST be deterministic given (step.params, step.input_ids, graph
contents). Non-determinism breaks Assumption 1 (replayable sound verification).

Handlers we ship:
    - identity / concat        (already in pcg.checker.build_default_replayer)
    - bm25_retrieve_replay     (recompute BM25 over committed passages)
    - span_extract             (regex-based span extraction)
    - nli_filter               (deterministic textual-entailment filter)
    - schema_validate          (jsonschema validation)
"""
from __future__ import annotations

import json
import re

from pcg.certificate import ReplayableStep
from pcg.checker import CompositeReplayer, build_default_replayer
from pcg.graph import AgenticRuntimeGraph, MaskedGraph, TruthNode

GraphLike = AgenticRuntimeGraph | MaskedGraph


# ---------------------------------------------------------------------------
# bm25_retrieve_replay
# ---------------------------------------------------------------------------


def bm25_retrieve_replay(step: ReplayableStep, graph: GraphLike) -> bytes:
    """Replay BM25 retrieval deterministically over committed evidence.

    Inputs:
        step.params:
            query:    the query string the Prover used
            top_k:    how many passages to return
            candidate_ids: list of TruthNode ids forming the candidate pool
        step.input_ids: ignored (the candidate pool comes from params)

    Output:
        UTF-8 bytes containing a JSON list of selected passage ids in BM25-rank
        order. The Verifier compares hash(this output) against
        step.output_digest. A mismatch (e.g., from index drift) is a replay
        failure, surfacing as ReplayFail in Theorem 1.
    """
    from pcg.retrieval import BM25Index

    query: str = step.params["query"]
    top_k: int = int(step.params.get("top_k", 5))
    candidate_ids: list[str] = list(step.params.get("candidate_ids", []))

    from pcg.datasets.base import EvidenceItem
    items = []
    for nid in candidate_ids:
        n = graph.nodes.get(nid)
        if isinstance(n, TruthNode):
            items.append(EvidenceItem(
                id=nid,
                title=n.attr.get("title", ""),
                text=n.payload.decode("utf-8", errors="replace"),
            ))
    idx = BM25Index.build(items)
    hits = idx.search(query, top_k=top_k)
    out_ids = [h[0].id for h in hits]   # search returns (item, score) tuples
    return json.dumps(out_ids).encode("utf-8")


# ---------------------------------------------------------------------------
# span_extract — pull out a span from concatenated evidence
# ---------------------------------------------------------------------------


def span_extract(step: ReplayableStep, graph: GraphLike) -> bytes:
    """Concatenate input TruthNode payloads, then extract the first span
    matching `step.params["pattern"]` (a regex). Returns the matched text
    in UTF-8 bytes.

    If no match: returns b"". (The entailment check will reject downstream.)
    """
    pattern: str = step.params["pattern"]
    flags = re.IGNORECASE if step.params.get("case_insensitive", True) else 0
    parts: list[str] = []
    for nid in step.input_ids:
        n = graph.nodes.get(nid)
        if isinstance(n, TruthNode):
            parts.append(n.payload.decode("utf-8", errors="replace"))
    blob = "\n".join(parts)
    m = re.search(pattern, blob, flags=flags)
    return m.group(0).encode("utf-8") if m else b""


# ---------------------------------------------------------------------------
# nli_filter — deterministic textual-entailment filter
# ---------------------------------------------------------------------------


def nli_filter(step: ReplayableStep, graph: GraphLike) -> bytes:
    """A deterministic NLI-style filter that does NOT require a model.

    Implements: claim is entailed by premise iff every content word in claim
    appears in premise (after lowercasing + stop-word removal). This is much
    weaker than a real NLI model but it is sound: it never says "entail" when
    the claim has content the premise lacks, so it preserves Assumption 2's
    soundness chain.

    For experiments using a real NLI model (e.g., `roberta-large-mnli`), the
    Prover should still record the model snapshot in step.params["model_sha"]
    so the Verifier can fail loud if a model swap silently changes the output.
    """
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "at", "for", "with", "by", "from", "and",
        "or", "but", "if", "then", "than", "that", "which", "this", "these",
        "those", "it", "its", "as", "do", "does", "did", "have", "has", "had",
    }
    claim: str = step.params["claim"]
    premise_parts = []
    for nid in step.input_ids:
        n = graph.nodes.get(nid)
        if isinstance(n, TruthNode):
            premise_parts.append(n.payload.decode("utf-8", errors="replace"))
    premise = " ".join(premise_parts).lower()
    claim_words = {
        w for w in re.findall(r"\b\w+\b", claim.lower()) if w not in stopwords
    }
    premise_words = set(re.findall(r"\b\w+\b", premise))
    accepted = bool(claim_words) and claim_words.issubset(premise_words)
    return (claim if accepted else "").encode("utf-8")


# ---------------------------------------------------------------------------
# schema_validate
# ---------------------------------------------------------------------------


def schema_validate(step: ReplayableStep, graph: GraphLike) -> bytes:
    """Validate input JSON bytes against the schema in step.params["schema"].

    Returns the input bytes unchanged if valid; b"" if invalid. Always
    deterministic since `jsonschema.validate` is pure.
    """
    try:
        import jsonschema
    except ImportError:
        # If jsonschema is missing we fail closed: empty output causes
        # entailment rejection downstream (and we get a clear error in logs).
        return b""

    schema = step.params["schema"]
    parts = []
    for nid in step.input_ids:
        n = graph.nodes.get(nid)
        if isinstance(n, TruthNode):
            parts.append(n.payload.decode("utf-8", errors="replace"))
    blob = "\n".join(parts)
    try:
        instance = json.loads(blob)
        jsonschema.validate(instance=instance, schema=schema)
        return blob.encode("utf-8")
    except (json.JSONDecodeError, jsonschema.ValidationError):
        return b""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_pcg_replayer() -> CompositeReplayer:
    """Replayer with all PCG-MAS handlers wired up.

    Use this at experiment time. The default replayer (no specialized ops)
    is used by the theory-only unit tests where pipelines are minimal.
    """
    rep = build_default_replayer()
    rep.register("bm25_retrieve_replay", bm25_retrieve_replay)
    rep.register("span_extract", span_extract)
    rep.register("nli_filter", nli_filter)
    rep.register("schema_validate", schema_validate)
    return rep


# Alias used by pcg.orchestrator.__init__
build_replayer_with_handlers = build_pcg_replayer
