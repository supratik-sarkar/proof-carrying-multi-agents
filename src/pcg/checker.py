"""
Checker — the deterministic external verifier.

Implements Definitions 2.9, 2.10 and the unified Check predicate in Eq. (6).
The Checker is a *pure function* of (certificate, graph): it holds no state
that isn't derived from its inputs, and it makes no external calls (tools are
replayed from committed outputs, see Appendix A.2 "Replay mode").

Check(Z; G_t) = Check_clm(Z^clm; G_t) AND Check_exe(Z^exe)
where
    Check_clm verifies:
        (1) hash commitment for every v in S: H(x(v)) == h(v)
        (2) pipeline replay: y' = Pi({x(v)}; meta), H(y') == y_digest
        (3) entailment: Check_vdash(y, c; R) == 1
    Check_exe verifies:
        (1) every logged tool call name is in Gamma.tool_allowlist
        (2) memory accesses respect Gamma.memory_access
        (3) delegations are in Gamma.allowed_delegations
        (4) required schemas/policies are attached to the run
        (5) resource budgets are respected

The entailment check is pluggable: the framework separates soundness (rule
system R, Assumption 3.2) from the concrete checker implementation. By
default we provide two implementations:

    - `ExactMatchEntailment`: for QA tasks where the answer string must appear
      verbatim in the replay output (strict soundness, easy to audit).
    - `NLIEntailment`: wraps an HF NLI model as a FILTER on top of an
      exact-match prefix (soundness is grounded in the exact-match part; NLI
      only adds completeness). This addresses the theory concern that
      ML-based entailment breaks Assumption 3.2 if used alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pcg.certificate import (
    ClaimCertificate,
    ExecutionCertificate,
    GroundingCertificate,
    ReplayableStep,
)
from pcg.commitments import H, verify
from pcg.graph import (
    AgenticRuntimeGraph,
    DelegationEdge,
    MaskedGraph,
    MemoryNode,
    NodeType,
    PolicyNode,
    SchemaNode,
    ToolCallNode,
    TruthNode,
)

GraphLike = AgenticRuntimeGraph | MaskedGraph


# -----------------------------------------------------------------------------
# Entailment interface
# -----------------------------------------------------------------------------


class EntailmentChecker(Protocol):
    """Check_vdash(y, c; R) - the deterministic entailment checker.

    Must be deterministic given (y, c). May depend on a frozen rule system R
    that is provided at construction time.
    """

    def check(self, y: Any, claim_text: str) -> bool: ...


@dataclass
class ExactMatchEntailment:
    """Check_vdash by verbatim span match.

    Sound by construction: if the claim string is a substring of the replay
    output after canonicalization, then the replay output entails the claim
    under the "verbatim substring" rule system R. This is what we use for
    the paper's main results on HotpotQA / 2Wiki where answers are short.
    """

    case_insensitive: bool = True

    def check(self, y: Any, claim_text: str) -> bool:
        y_str = y if isinstance(y, str) else str(y)
        if self.case_insensitive:
            return claim_text.lower().strip() in y_str.lower()
        return claim_text.strip() in y_str


@dataclass
class NLIEntailment:
    """Sound-with-filter: run an NLI model as a filter, but only accept when
    an underlying `base` check also accepts.

    This keeps Assumption 3.2 (sound entailment) pinned to the `base` check
    (typically ExactMatchEntailment). NLI can only REDUCE completeness by
    rejecting base-accepted pairs; it can never produce a false accept that
    the base check would have rejected. Formally:

        NLIEntailment.check(y, c) = base.check(y, c) AND nli_fn(y, c)

    so acceptance implies base.check == 1.
    """

    base: EntailmentChecker
    nli_fn: Callable[[str, str], bool]

    def check(self, y: Any, claim_text: str) -> bool:
        if not self.base.check(y, claim_text):
            return False
        y_str = y if isinstance(y, str) else str(y)
        return self.nli_fn(y_str, claim_text)


# -----------------------------------------------------------------------------
# Pipeline replay
# -----------------------------------------------------------------------------


class Replayer(Protocol):
    """Given a pipeline step and the graph, produce the step's output.

    Implementations must be deterministic: the same `step` and the same
    committed inputs must produce bit-identical output across runs.
    """

    def run(self, step: ReplayableStep, graph: GraphLike) -> bytes: ...


@dataclass
class CompositeReplayer:
    """Dispatches replay to per-op implementations.

    The user registers a function `fn(step, graph) -> bytes` for each op name.
    The replayer concatenates outputs across steps to form the final y; this
    matches the convention that the last step's output is the one entailment
    is checked against, but makes all intermediate digests available.
    """

    handlers: dict[str, Callable[[ReplayableStep, GraphLike], bytes]] = field(default_factory=dict)

    def register(self, op: str, fn: Callable[[ReplayableStep, GraphLike], bytes]) -> None:
        self.handlers[op] = fn

    def run(self, step: ReplayableStep, graph: GraphLike) -> bytes:
        if step.op not in self.handlers:
            raise KeyError(
                f"No replay handler for op '{step.op}'. "
                f"Registered: {sorted(self.handlers)}"
            )
        return self.handlers[step.op](step, graph)


# -----------------------------------------------------------------------------
# CheckResult
# -----------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Structured result from Checker.check(...).

    Importantly, `passed` is a single bit (Eq. 6), but we also record
    which channel failed (integrity, replay, entailment, execution) so that
    downstream evaluation can estimate the audit-decomposition terms of
    Theorem 1 (Eq. 7).
    """

    passed: bool
    integrity_ok: bool = True
    replay_ok: bool = True
    entailment_ok: bool = True
    execution_ok: bool = True
    # Detailed reasons; empty if passed.
    reasons: list[str] = field(default_factory=list)
    # Derived data useful for downstream analysis.
    replay_output: bytes | None = None
    replay_output_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "integrity_ok": self.integrity_ok,
            "replay_ok": self.replay_ok,
            "entailment_ok": self.entailment_ok,
            "execution_ok": self.execution_ok,
            "reasons": list(self.reasons),
            "replay_output_digest": self.replay_output_digest,
        }


# -----------------------------------------------------------------------------
# Checker
# -----------------------------------------------------------------------------


@dataclass
class Checker:
    """External, deterministic verifier for unified grounding certificates.

    A Checker is stateless: you can call `check(...)` from multiple threads
    on the same instance, provided the `entailment` and `replayer` components
    are themselves thread-safe.
    """

    entailment: EntailmentChecker
    replayer: Replayer

    # ---- public API ----
    def check(self, cert: GroundingCertificate, graph: GraphLike) -> CheckResult:
        """Returns a CheckResult implementing Eq. (6)."""
        r = CheckResult(passed=False)

        # --- claim-side: Eq. (5a) ---
        clm_ok, clm_y = self._check_claim_side(cert.claim_cert, graph, r)

        # --- execution-side: Eq. (5b) ---
        exe_ok = self._check_execution_side(cert.exec_cert, graph, r)

        r.passed = clm_ok and exe_ok
        return r

    # ---- internals ----
    def _check_claim_side(
        self,
        cc: ClaimCertificate,
        graph: GraphLike,
        r: CheckResult,
    ) -> tuple[bool, bytes | None]:
        # (1) Hash commitments
        nodes = graph.nodes
        for eid, expected_digest in zip(cc.evidence_ids, cc.evidence_digests):
            if eid not in nodes:
                r.integrity_ok = False
                r.reasons.append(f"evidence_node_missing:{eid}")
                continue
            node = nodes[eid]
            if not isinstance(node, TruthNode):
                r.integrity_ok = False
                r.reasons.append(f"evidence_not_truth:{eid}:{type(node).__name__}")
                continue
            actual = H(node.content_for_hash())
            if actual != expected_digest:
                r.integrity_ok = False
                r.reasons.append(f"hash_mismatch:{eid}")

        if not r.integrity_ok:
            return False, None

        # (2) Pipeline replay: sequentially run each step, accumulate output.
        #     The pipeline's last step output is what entailment is checked against.
        try:
            last_output: bytes = b""
            for step in cc.pipeline:
                last_output = self.replayer.run(step, graph)
            y_digest = H(last_output)
            r.replay_output = last_output
            r.replay_output_digest = y_digest
        except Exception as exc:   # noqa: BLE001 — we surface all replay errors uniformly
            r.replay_ok = False
            r.reasons.append(f"replay_exception:{type(exc).__name__}:{exc}")
            return False, None

        if y_digest != cc.replay_output_digest:
            r.replay_ok = False
            r.reasons.append(
                f"replay_digest_mismatch:expected={cc.replay_output_digest},got={y_digest}"
            )
            return False, last_output

        # (3) Entailment: Check_vdash(y, c; R) == 1
        claim_node = nodes.get(cc.claim_id)
        claim_text = getattr(claim_node, "canonical", "") or getattr(claim_node, "raw", "")
        if not claim_text:
            r.entailment_ok = False
            r.reasons.append(f"claim_text_missing:{cc.claim_id}")
            return False, last_output

        if not self.entailment.check(last_output.decode("utf-8", errors="replace"), claim_text):
            r.entailment_ok = False
            r.reasons.append(
                f"entailment_rejected:claim={claim_text[:80]!r}"
            )
            return False, last_output

        return True, last_output

    def _check_execution_side(
        self,
        ec: ExecutionCertificate,
        graph: GraphLike,
        r: CheckResult,
    ) -> bool:
        contract = ec.contract
        nodes = graph.nodes

        # (a) Tool allowlist
        tool_call_count = 0
        for tid in ec.tool_call_ids:
            node = nodes.get(tid)
            if not isinstance(node, ToolCallNode):
                r.execution_ok = False
                r.reasons.append(f"tool_node_missing_or_wrong_type:{tid}")
                continue
            if contract.tool_allowlist and node.tool_name not in contract.tool_allowlist:
                r.execution_ok = False
                r.reasons.append(f"tool_not_allowed:{node.tool_name}")
            tool_call_count += 1
        if contract.max_tool_calls is not None and tool_call_count > contract.max_tool_calls:
            r.execution_ok = False
            r.reasons.append(
                f"tool_budget_exceeded:{tool_call_count}>{contract.max_tool_calls}"
            )

        # (b) Memory-access constraints
        for mid in ec.memory_node_ids:
            node = nodes.get(mid)
            if not isinstance(node, MemoryNode):
                r.execution_ok = False
                r.reasons.append(f"mem_node_missing_or_wrong_type:{mid}")
                continue
            key = f"{node.scope}:{node.op}"
            if contract.memory_access and key not in contract.memory_access:
                r.execution_ok = False
                r.reasons.append(f"memory_access_violation:{key}")

        # (c) Delegation constraints
        for did in ec.delegation_ids:
            node = nodes.get(did)
            if not isinstance(node, DelegationEdge):
                r.execution_ok = False
                r.reasons.append(f"delegation_node_missing_or_wrong_type:{did}")
                continue
            if contract.allowed_delegations and node.child_agent not in contract.allowed_delegations:
                r.execution_ok = False
                r.reasons.append(f"delegation_not_allowed:{node.child_agent}")

        # (d) Required schemas present
        attached_schema_ids = {
            nodes[s].schema_id
            for s in ec.schema_node_ids
            if isinstance(nodes.get(s), SchemaNode)
        }
        missing_schemas = contract.required_schema_ids - attached_schema_ids
        if missing_schemas:
            r.execution_ok = False
            r.reasons.append(f"schema_missing:{sorted(missing_schemas)}")

        # (e) Required policies present
        attached_policy_ids = {
            nodes[p].policy_id
            for p in ec.policy_node_ids
            if isinstance(nodes.get(p), PolicyNode)
        }
        missing_policies = contract.required_policy_ids - attached_policy_ids
        if missing_policies:
            r.execution_ok = False
            r.reasons.append(f"policy_missing:{sorted(missing_policies)}")

        return r.execution_ok


# -----------------------------------------------------------------------------
# Convenience: a default replayer registry covering the ops we use in R1-R5
# -----------------------------------------------------------------------------


def build_default_replayer() -> CompositeReplayer:
    """Return a CompositeReplayer wired up for the ops used by our experiments.

    Kept in-module so the Checker is a one-stop shop; real handlers live in
    `pcg.orchestrator.replay_handlers` and are registered there.
    """
    rep = CompositeReplayer()

    # 'identity' replay: passes through the payload of the first input TruthNode.
    # Useful for minimal pipelines where the "aligned evidence" is just the raw
    # evidence text itself.
    def _identity(step: ReplayableStep, graph: GraphLike) -> bytes:
        if not step.input_ids:
            return b""
        first = graph.nodes.get(step.input_ids[0])
        if isinstance(first, TruthNode):
            return first.payload
        return b""

    # 'concat' replay: byte-concatenate payloads of all input TruthNodes with a
    # canonical delimiter. Deterministic across runs.
    def _concat(step: ReplayableStep, graph: GraphLike) -> bytes:
        delim = step.params.get("delim", "\n").encode("utf-8")
        chunks: list[bytes] = []
        for nid in step.input_ids:
            n = graph.nodes.get(nid)
            if isinstance(n, TruthNode):
                chunks.append(n.payload)
        return delim.join(chunks)

    rep.register("identity", _identity)
    rep.register("concat", _concat)
    return rep
