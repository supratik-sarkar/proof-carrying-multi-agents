"""
Agentic runtime graph (Definition 2.1, Eq. 1 of the paper).

This module defines the typed evidence graph G_t = (V_t, E_t, tau_V, tau_E, attr_t)
that serves as the *external audit state* shared by agents and the verifier.

Design notes:
    - All node types inherit from `GraphNode` so that verification / hashing logic can
      be uniform. Type information (`NodeType`) is carried alongside the dataclass
      type so that polymorphic verification (Check_clm, Check_exe) can branch on it.
    - The graph is append-only from the perspective of verification: masking
      (used by `pcg.responsibility`) is implemented as a *view* that filters edges,
      never an in-place mutation. This is what makes replay-based interventions
      well-defined (see Appendix A.2 of the paper: "Replay mode").
    - Every node carries `run_id` and `ts` so the same graph object can hold multiple
      episodes — we rely on this in R5 (token overhead) to compare branches.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

# -----------------------------------------------------------------------------
# Node / edge types (paper Section 3, Definition 2.1 enumeration)
# -----------------------------------------------------------------------------


class NodeType(str, Enum):
    """tau_V: node-type map from Definition 2.1.

    The enum values are stable strings so that serialized graphs can round-trip
    across processes and across Python versions.
    """

    TRUTH = "truth"             # V_t^truth: immutable evidence-bearing nodes
    SOURCE = "source"           # provenance descriptor (URL, DOI, dataset ID, ...)
    TOOL = "tool"               # tool-call descriptor (name, version, args, endpoint)
    SCHEMA = "schema"           # validator contract for structured execution
    MEMORY = "memory"           # short-term, persistent, or episodic memory
    POLICY = "policy"           # guardrail / constitution / tool allow-list
    MESSAGE = "message"         # inter-agent or agent-to-tool message
    ACTION = "action"           # retrieve/parse/verify/cite/escalate/refuse/answer
    DELEGATION = "delegation"   # sub-agent assignment and returned artifact
    CLAIM = "claim"             # canonical claim node for c in C


class EdgeType(str, Enum):
    """tau_E: edge-type map from Definition 2.1."""

    RETRIEVED_FROM = "retrieved_from"
    PRODUCED_BY_TOOL = "produced_by_tool"
    PARSED_TO = "parsed_to"
    ALIGNED_TO = "aligned_to"
    CITES = "cites"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DELEGATED_TO = "delegated_to"       # parent agent -> sub-agent
    VALIDATED_BY = "validated_by"       # action -> schema / policy node
    CAUSES = "causes"                   # causal provenance (optional, often inferred)


# -----------------------------------------------------------------------------
# Node dataclasses
# -----------------------------------------------------------------------------


def _new_id() -> str:
    """Canonical unique ID. Kept short so log dumps are readable."""
    return uuid.uuid4().hex[:16]


def _now() -> float:
    return time.time()


@dataclass
class GraphNode:
    """Base class. Concrete node types specialize this with payload / attributes.

    Invariants (enforced by `AgenticRuntimeGraph.add_node`):
        - `id` is globally unique within the graph.
        - `type` matches the concrete subclass.
        - `run_id` identifies the logical execution / episode.
    """

    id: str = field(default_factory=_new_id)
    type: NodeType = NodeType.ACTION  # overridden by subclasses
    run_id: str = ""                  # set by AgenticRuntimeGraph
    ts: float = field(default_factory=_now)
    attr: dict[str, Any] = field(default_factory=dict)

    def content_for_hash(self) -> bytes:
        """Canonical byte representation used by `pcg.commitments.H(.)`.

        Subclasses override this. We use byte-level canonicalization (not JSON
        dumps) to avoid serialization drift across library versions — see
        Appendix A.1 "Canonical IDs" of the paper.
        """
        raise NotImplementedError


@dataclass
class TruthNode(GraphNode):
    """Immutable evidence-bearing node. Stores payload x(v) and commitment h(v).

    The payload is `bytes` to avoid encoding ambiguity; higher layers (dataset
    loaders) are responsible for declaring the content type via `attr["mime"]`.
    """

    payload: bytes = b""
    mime: str = "text/plain"
    source_id: str | None = None   # foreign key to a SOURCE node if present
    type: NodeType = NodeType.TRUTH

    def content_for_hash(self) -> bytes:
        # Prefix with mime so hashes distinguish "the same bytes" under different encodings.
        return self.mime.encode("utf-8") + b"\x00" + self.payload


@dataclass
class SourceNode(GraphNode):
    """Provenance descriptor: URL, dataset ID, publisher, etc."""

    url: str | None = None
    authority_id: str | None = None     # e.g., DOI prefix, journal, standards body
    publisher_id: str | None = None
    domain: str | None = None           # eTLD+1
    type: NodeType = NodeType.SOURCE

    def content_for_hash(self) -> bytes:
        parts = [
            self.url or "",
            self.authority_id or "",
            self.publisher_id or "",
            self.domain or "",
        ]
        return "\x00".join(parts).encode("utf-8")


@dataclass
class ToolCallNode(GraphNode):
    """Tool / function / MCP invocation. This is the node that makes the
    framework *agent-specific*: Check_exe validates these against the execution
    contract Gamma. Without this, the framework degenerates to RAG verification.
    """

    tool_name: str = ""
    tool_version: str = ""
    endpoint: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    # Tool output bytes; these become a Truth node when committed (the checker
    # does not trust live tool calls — see Appendix A.2 "Tool replay vs fresh mode").
    output_digest: str | None = None  # hash of the output Truth node, if any
    latency_ms: float = 0.0
    type: NodeType = NodeType.TOOL

    def content_for_hash(self) -> bytes:
        # JSON-canonical key-sorted dump (lazy import to keep core module light)
        import json
        args_blob = json.dumps(self.args, sort_keys=True, separators=(",", ":"))
        parts = [
            self.tool_name,
            self.tool_version,
            self.endpoint or "",
            args_blob,
            self.output_digest or "",
        ]
        return "\x00".join(parts).encode("utf-8")


@dataclass
class SchemaNode(GraphNode):
    """JSON-Schema-like validator for structured tool outputs or messages."""

    schema_id: str = ""
    schema_version: str = ""
    schema_dict: dict[str, Any] = field(default_factory=dict)
    type: NodeType = NodeType.SCHEMA

    def content_for_hash(self) -> bytes:
        import json
        return json.dumps(
            {"id": self.schema_id, "v": self.schema_version, "s": self.schema_dict},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


@dataclass
class MemoryNode(GraphNode):
    """Short-term / persistent / episodic memory read or write."""

    memory_id: str = ""
    scope: str = "short_term"      # short_term | persistent | episodic
    op: str = "read"               # read | write
    key: str = ""
    value_digest: str | None = None   # hash of value (actual value is a Truth node)
    type: NodeType = NodeType.MEMORY

    def content_for_hash(self) -> bytes:
        return "\x00".join(
            [self.memory_id, self.scope, self.op, self.key, self.value_digest or ""]
        ).encode("utf-8")


@dataclass
class PolicyNode(GraphNode):
    """Guardrail, tool allow-list, or action constraint clause."""

    policy_id: str = ""
    clause_id: str = ""
    kind: str = "tool_allow"       # tool_allow | memory_access | action_constraint | guardrail
    content: str = ""              # human-readable clause text (exact source of truth)
    type: NodeType = NodeType.POLICY

    def content_for_hash(self) -> bytes:
        return "\x00".join(
            [self.policy_id, self.clause_id, self.kind, self.content]
        ).encode("utf-8")


@dataclass
class MessageNode(GraphNode):
    """Inter-agent or agent-to-tool message."""

    from_agent: str = ""
    to_agent: str = ""
    role: str = "user"             # user | assistant | system | tool | agent
    content: str = ""
    n_tokens: int = 0              # USED BY THE OVERHEAD METER (R5)
    type: NodeType = NodeType.MESSAGE

    def content_for_hash(self) -> bytes:
        return "\x00".join(
            [self.from_agent, self.to_agent, self.role, self.content]
        ).encode("utf-8")


@dataclass
class ActionNode(GraphNode):
    """A discrete agent action: retrieve, parse, cite, escalate, refuse, answer."""

    action: str = ""               # retrieve | parse | cite | verify | escalate | refuse | answer
    agent_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    type: NodeType = NodeType.ACTION

    def content_for_hash(self) -> bytes:
        import json
        return (
            self.action
            + "\x00"
            + self.agent_id
            + "\x00"
            + json.dumps(self.args, sort_keys=True, separators=(",", ":"))
        ).encode("utf-8")


@dataclass
class DelegationEdge(GraphNode):
    """A delegation event: parent agent assigns a sub-task to a child agent.

    Modeled as a node (not an edge) so that `Resp_del(e_del)` in Definition 2.13
    refers to a concrete, maskable artifact. See `pcg.responsibility`.
    """

    parent_agent: str = ""
    child_agent: str = ""
    task_description: str = ""
    returned_artifact_digest: str | None = None
    type: NodeType = NodeType.DELEGATION

    def content_for_hash(self) -> bytes:
        return "\x00".join(
            [
                self.parent_agent,
                self.child_agent,
                self.task_description,
                self.returned_artifact_digest or "",
            ]
        ).encode("utf-8")


@dataclass
class ClaimNode(GraphNode):
    """Canonical claim node representing c in C.

    Stores both raw and canonicalized forms so that the entailment checker
    operates on a deterministic normalization (see Appendix A.7 of the paper).
    """

    raw: str = ""
    canonical: str = ""     # after Unicode NFKC + whitespace + locale-invariant normalization
    claim_type: str = "text"    # text | entity_relation_triple | typed_kv
    type: NodeType = NodeType.CLAIM

    def content_for_hash(self) -> bytes:
        return (self.claim_type + "\x00" + self.canonical).encode("utf-8")


# -----------------------------------------------------------------------------
# Edges
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """A typed directed edge between two nodes."""

    src: str     # node id
    dst: str     # node id
    type: EdgeType


# -----------------------------------------------------------------------------
# The runtime graph
# -----------------------------------------------------------------------------


class AgenticRuntimeGraph:
    """G_t = (V_t, E_t, tau_V, tau_E, attr_t) from Definition 2.1.

    The graph is append-only in the abstract; masking is a read-only view.
    We index nodes by id and maintain a type-indexed secondary map for fast
    retrieval during verification and independence checks.

    Determinism: iteration order over nodes is insertion order (CPython dict
    semantics), which matters because some checker steps are order-sensitive.
    """

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or _new_id()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[Edge] = []
        self._nodes_by_type: dict[NodeType, list[str]] = {t: [] for t in NodeType}
        # Reverse adjacency for mask-based interventions.
        self._out_edges: dict[str, list[Edge]] = {}
        self._in_edges: dict[str, list[Edge]] = {}

    # ---- node/edge mutators ----
    def add_node(self, node: GraphNode) -> str:
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        if not node.run_id:
            node.run_id = self.run_id
        self._nodes[node.id] = node
        self._nodes_by_type[node.type].append(node.id)
        self._out_edges.setdefault(node.id, [])
        self._in_edges.setdefault(node.id, [])
        return node.id

    def add_edge(self, src: str, dst: str, etype: EdgeType) -> None:
        if src not in self._nodes or dst not in self._nodes:
            raise KeyError(f"Edge references unknown node: {src} -> {dst}")
        e = Edge(src=src, dst=dst, type=etype)
        self._edges.append(e)
        self._out_edges[src].append(e)
        self._in_edges[dst].append(e)

    # ---- accessors ----
    @property
    def nodes(self) -> dict[str, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> list[Edge]:
        return self._edges

    def nodes_of_type(self, t: NodeType) -> list[GraphNode]:
        return [self._nodes[nid] for nid in self._nodes_by_type[t]]

    def out_edges(self, nid: str) -> list[Edge]:
        return self._out_edges.get(nid, [])

    def in_edges(self, nid: str) -> list[Edge]:
        return self._in_edges.get(nid, [])

    def truth_nodes(self) -> list[TruthNode]:
        """V_t^truth, the evidence-bearing truth nodes."""
        return [n for n in self.nodes_of_type(NodeType.TRUTH) if isinstance(n, TruthNode)]

    # ---- iteration ----
    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[GraphNode]:
        return iter(self._nodes.values())

    def __contains__(self, nid: str) -> bool:
        return nid in self._nodes

    # ---- masking (read-only view) ----
    def mask(self, node_ids: set[str]) -> "MaskedGraph":
        """Return a view of the graph with `node_ids` masked out.

        This realizes G_t^{\\setminus e} from Definition 2.12 (Eq. 6) for a
        single- or multi-component mask. The returned view supports the same
        accessors as `AgenticRuntimeGraph` but hides masked nodes and their
        incident edges. Used by `pcg.responsibility`.
        """
        return MaskedGraph(base=self, masked=frozenset(node_ids))


class MaskedGraph:
    """Read-only view of an AgenticRuntimeGraph with a subset of nodes masked.

    Implements the same `nodes_of_type` / `out_edges` / `in_edges` / `truth_nodes`
    accessors as the base graph, so checker code can accept either.
    """

    def __init__(self, base: AgenticRuntimeGraph, masked: frozenset[str]) -> None:
        self._base = base
        self._masked = masked
        self.run_id = base.run_id

    @property
    def masked(self) -> frozenset[str]:
        return self._masked

    @property
    def nodes(self) -> dict[str, GraphNode]:
        return {nid: n for nid, n in self._base.nodes.items() if nid not in self._masked}

    @property
    def edges(self) -> list[Edge]:
        return [
            e for e in self._base.edges
            if e.src not in self._masked and e.dst not in self._masked
        ]

    def nodes_of_type(self, t: NodeType) -> list[GraphNode]:
        return [n for n in self._base.nodes_of_type(t) if n.id not in self._masked]

    def out_edges(self, nid: str) -> list[Edge]:
        if nid in self._masked:
            return []
        return [e for e in self._base.out_edges(nid) if e.dst not in self._masked]

    def in_edges(self, nid: str) -> list[Edge]:
        if nid in self._masked:
            return []
        return [e for e in self._base.in_edges(nid) if e.src not in self._masked]

    def truth_nodes(self) -> list[TruthNode]:
        return [n for n in self._base.truth_nodes() if n.id not in self._masked]

    def __contains__(self, nid: str) -> bool:
        return nid in self._base and nid not in self._masked

    def __len__(self) -> int:
        return len(self._base) - len(self._masked)
