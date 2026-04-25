"""
(delta, kappa)-independence of support paths (Definition 2.11).

Also hosts the operational proxies from Appendix A.4: provenance labels,
tool overlap, and shingle-based replayable overlap.

This module is the load-bearing theoretical bridge between the paper's
*operational* independence notion (Def 2.11) and the *probabilistic*
residual-dependence factor rho used in Theorem 2 (Assumption 3). Specifically:

    - `path_overlap(pi_i, pi_j)` returns a deterministic, replayable score
      in [0, 1] that the verifier can recompute from logs alone. This is
      the Overlap(pi_i, pi_j) function in Eq. (19).
    - `are_independent(pi_i, pi_j, delta, kappa)` is the boolean predicate
      of Definition 2.11.
    - `rho_upper_bound(...)` (in pcg.eval.rho_estimator) uses the family
      of independent paths produced here to give a statistically-valid
      upper confidence bound on rho (addresses the theoretical concern
      that rho is otherwise defined tautologically).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from pcg.graph import (
    AgenticRuntimeGraph,
    Edge,
    EdgeType,
    NodeType,
    SourceNode,
    ToolCallNode,
    TruthNode,
)


# -----------------------------------------------------------------------------
# SupportPath: a realized evidence path in G_t (Def 2.11 preamble)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportPath:
    """A realized directed path from a TruthNode to the ClaimNode.

    The path is represented by its *node IDs in order*; edges between
    consecutive nodes are implicit (the graph supplies them). We also
    precompute the set of tool-step IDs seen along the path so that
    ToolOverlap is an O(min(|a|, |b|)) set operation.
    """

    node_ids: tuple[str, ...]
    tool_step_ids: frozenset[str] = frozenset()

    @property
    def head(self) -> str:
        return self.node_ids[0]

    @property
    def tail(self) -> str:
        return self.node_ids[-1]


def extract_support_path(
    graph: AgenticRuntimeGraph,
    truth_id: str,
    claim_id: str,
    follow_edge_types: Iterable[EdgeType] | None = None,
) -> SupportPath | None:
    """BFS from a TruthNode to the ClaimNode following SUPPORTS / CITES /
    ALIGNED_TO / PARSED_TO / RETRIEVED_FROM edges (reversed where needed).

    Returns None if no path exists. The BFS visits each node at most once
    so multiple independent paths through the same TruthNode will each be
    returned if extracted via different truth_id seeds.
    """
    allow = (
        set(follow_edge_types)
        if follow_edge_types is not None
        else {EdgeType.SUPPORTS, EdgeType.CITES, EdgeType.ALIGNED_TO,
              EdgeType.PARSED_TO, EdgeType.RETRIEVED_FROM, EdgeType.PRODUCED_BY_TOOL}
    )
    visited = {truth_id}
    # Each entry: (current_node_id, path_so_far, tool_steps_so_far)
    queue: list[tuple[str, tuple[str, ...], frozenset[str]]] = [
        (truth_id, (truth_id,), frozenset())
    ]
    while queue:
        nid, path, tools = queue.pop(0)
        if nid == claim_id:
            return SupportPath(node_ids=path, tool_step_ids=tools)
        for e in graph.out_edges(nid):
            if e.type not in allow:
                continue
            if e.dst in visited:
                continue
            visited.add(e.dst)
            new_tools = tools
            dst_node = graph.nodes[e.dst]
            if isinstance(dst_node, ToolCallNode):
                new_tools = frozenset({*tools, e.dst})
            queue.append((e.dst, (*path, e.dst), new_tools))
    return None


# -----------------------------------------------------------------------------
# Provenance classes (Appendix A.4, Eq. 16)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceLabel:
    """Prov(v) := (AuthorityID, PublisherID, Domain).

    The two paths are provenance-disjoint iff the AuthorityID (strict),
    PublisherID (moderate), or Domain (weakest) fields differ. The strictness
    level is configurable at the call site.
    """

    authority_id: str | None
    publisher_id: str | None
    domain: str | None

    def is_empty(self) -> bool:
        return not any([self.authority_id, self.publisher_id, self.domain])


def provenance_of(graph: AgenticRuntimeGraph, truth_id: str) -> ProvenanceLabel:
    """Follow `retrieved_from` edges backward from the TruthNode to find its
    SourceNode, and return the provenance label.

    If no SourceNode is found, returns an empty label (all fields None).
    """
    node = graph.nodes.get(truth_id)
    if node is None:
        return ProvenanceLabel(None, None, None)
    # The TruthNode may carry a direct source_id pointer (preferred).
    if isinstance(node, TruthNode) and node.source_id is not None:
        source = graph.nodes.get(node.source_id)
        if isinstance(source, SourceNode):
            return ProvenanceLabel(source.authority_id, source.publisher_id, source.domain)
    # Fallback: scan incoming RETRIEVED_FROM edges.
    for e in graph.in_edges(truth_id):
        if e.type == EdgeType.RETRIEVED_FROM:
            src = graph.nodes.get(e.src)
            if isinstance(src, SourceNode):
                return ProvenanceLabel(src.authority_id, src.publisher_id, src.domain)
    return ProvenanceLabel(None, None, None)


def provenance_disjoint(
    a: ProvenanceLabel,
    b: ProvenanceLabel,
    level: str = "authority",
) -> bool:
    """Return True iff the two labels are considered disjoint at `level`.

    level:
        "authority": strictest — authority IDs must differ and both non-None
        "publisher": publisher IDs must differ
        "domain":    domains must differ (weakest)
    """
    if level == "authority":
        return a.authority_id is not None and b.authority_id is not None \
            and a.authority_id != b.authority_id
    if level == "publisher":
        return a.publisher_id is not None and b.publisher_id is not None \
            and a.publisher_id != b.publisher_id
    if level == "domain":
        return a.domain is not None and b.domain is not None \
            and a.domain != b.domain
    raise ValueError(f"Unknown provenance level: {level}")


# -----------------------------------------------------------------------------
# Tool overlap (Appendix A.4, Eq. 17)
# -----------------------------------------------------------------------------


def tool_overlap(a: SupportPath, b: SupportPath) -> int:
    """|Tools(pi_a) cap Tools(pi_b)|, the count of shared tool-step IDs."""
    return len(a.tool_step_ids & b.tool_step_ids)


# -----------------------------------------------------------------------------
# Replayable shingle overlap (Appendix A.4, Eqs. 18-19)
# -----------------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")


def canonicalize_text(s: str) -> str:
    """The canonicalization rule used for shingle overlap.

    Applies Unicode NFKC, collapses whitespace, and lowercases. All three
    transformations are idempotent and deterministic across platforms.
    """
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def shingles(text: str, m: int = 5) -> frozenset[str]:
    """m-gram token shingles of canonicalized text. Deterministic."""
    tokens = canonicalize_text(text).split()
    if len(tokens) < m:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(" ".join(tokens[i : i + m]) for i in range(len(tokens) - m + 1))


def shingle_jaccard(text_a: str, text_b: str, m: int = 5) -> float:
    """Jaccard overlap of m-gram shingles. Returns 0.0 for empty inputs."""
    sa, sb = shingles(text_a, m=m), shingles(text_b, m=m)
    if not sa or not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def path_overlap(
    graph: AgenticRuntimeGraph,
    a: SupportPath,
    b: SupportPath,
    m: int = 5,
) -> float:
    """Overlap(pi_i, pi_j) from Eq. (19): Jaccard over m-gram shingles of the
    terminal evidence-text cited by each path.

    "Terminal" here means the TruthNode at the head of each path (the actual
    evidence item), not any intermediate transformation output. This is the
    definition that the verifier can reproduce from committed data alone,
    which is the whole point.
    """
    n_a = graph.nodes.get(a.head)
    n_b = graph.nodes.get(b.head)
    if not isinstance(n_a, TruthNode) or not isinstance(n_b, TruthNode):
        return 0.0
    text_a = n_a.payload.decode("utf-8", errors="replace")
    text_b = n_b.payload.decode("utf-8", errors="replace")
    return shingle_jaccard(text_a, text_b, m=m)


# -----------------------------------------------------------------------------
# (delta, kappa)-independence predicate
# -----------------------------------------------------------------------------


@dataclass
class IndependenceConfig:
    delta: float = 0.2
    kappa: int = 1
    provenance_level: str = "publisher"   # authority | publisher | domain
    shingle_m: int = 5


def are_independent(
    graph: AgenticRuntimeGraph,
    a: SupportPath,
    b: SupportPath,
    cfg: IndependenceConfig,
) -> bool:
    """The full predicate from Def 2.11, combining Eqs. (16)-(18)."""
    # Provenance disjointness
    prov_a = provenance_of(graph, a.head)
    prov_b = provenance_of(graph, b.head)
    if not provenance_disjoint(prov_a, prov_b, level=cfg.provenance_level):
        return False
    # Tool-step overlap
    if tool_overlap(a, b) >= cfg.kappa:
        return False
    # Shingle overlap
    if path_overlap(graph, a, b, m=cfg.shingle_m) > cfg.delta:
        return False
    return True


def filter_independent_family(
    graph: AgenticRuntimeGraph,
    paths: list[SupportPath],
    cfg: IndependenceConfig,
) -> list[SupportPath]:
    """Greedy construction of a maximal family of pairwise independent paths.

    We sweep paths in the given order and accept each one iff it is
    independent of every already-accepted path. The result is a family
    P_{delta,kappa} in the notation of Appendix A.4.

    Note: this is *greedy*, not optimal. Global maximum-independent-set is
    NP-hard; we keep the input order as the tie-breaker (usually sorted by
    Prover's quality estimate, so the best paths are kept first).
    """
    accepted: list[SupportPath] = []
    for p in paths:
        if all(are_independent(graph, p, q, cfg) for q in accepted):
            accepted.append(p)
    return accepted


# -----------------------------------------------------------------------------
# Semantic overlap (optional auditable variant, Appendix A.4 Eq. 20)
# -----------------------------------------------------------------------------


@dataclass
class SemanticOverlap:
    """Auditable semantic overlap: cos(phi(span_a), phi(span_b)) for a fixed
    public embedding model. Recorded by digest so the verifier can confirm.

    This is optional; `path_overlap` (shingle Jaccard) is the default because
    it requires no model dependency. When used, `embed_fn` is expected to be
    a deterministic function; the caller is responsible for declaring the
    model snapshot in the certificate's `meta` field so the verifier can
    reproduce it.
    """

    embed_fn: object                        # callable str -> np.ndarray
    alpha: float = 0.5                      # blend weight: alpha * Jaccard + (1-alpha) * sem
    shingle_m: int = 5

    def overlap(self, graph: AgenticRuntimeGraph, a: SupportPath, b: SupportPath) -> float:
        import numpy as np
        n_a = graph.nodes.get(a.head)
        n_b = graph.nodes.get(b.head)
        if not isinstance(n_a, TruthNode) or not isinstance(n_b, TruthNode):
            return 0.0
        text_a = n_a.payload.decode("utf-8", errors="replace")
        text_b = n_b.payload.decode("utf-8", errors="replace")
        jaccard = shingle_jaccard(text_a, text_b, m=self.shingle_m)
        emb_fn = self.embed_fn  # type: ignore[assignment]
        ea = np.asarray(emb_fn(text_a))     # type: ignore[operator]
        eb = np.asarray(emb_fn(text_b))     # type: ignore[operator]
        cos = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12))
        sem = max(0.0, cos)
        return self.alpha * jaccard + (1.0 - self.alpha) * sem
