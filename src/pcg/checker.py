"""Four-channel external verifier for PCG-MAS v4.

Implements the unified acceptance predicate

    Check(Z; G_t) = V_H * V_Pi * V_Gamma * V_entail

where:
    V_H       : evidence commitment / hash integrity
    V_Pi      : deterministic replay consistency
    V_Gamma   : execution-contract compliance
    V_entail  : claim entailment under a fixed rule system

The checker is intentionally deterministic and makes no live network/tool calls.
Tool outputs must already be committed as truth nodes in G_t.
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
from pcg.commitments import H
from pcg.graph import (
    AgenticRuntimeGraph,
    DelegationEdge,
    MaskedGraph,
    MemoryNode,
    PolicyNode,
    SchemaNode,
    ToolCallNode,
    TruthNode,
)

GraphLike = AgenticRuntimeGraph | MaskedGraph


# ---------------------------------------------------------------------------
# Entailment interface
# ---------------------------------------------------------------------------

class EntailmentChecker(Protocol):
    """Deterministic Check_vdash(y, c; R)."""

    def check(self, y: Any, claim_text: str) -> bool:
        ...


@dataclass
class ExactMatchEntailment:
    """Strict deterministic entailment via canonical substring matching."""

    case_insensitive: bool = True

    def check(self, y: Any, claim_text: str) -> bool:
        y_str = y if isinstance(y, str) else str(y)
        claim = claim_text.strip()
        if self.case_insensitive:
            return claim.lower() in y_str.lower()
        return claim in y_str


@dataclass
class NLIEntailment:
    """Sound-with-filter entailment.

    The learned NLI model can only reject a base-accepted pair; it cannot create
    an acceptance that the deterministic base checker rejected.
    """

    base: EntailmentChecker
    nli_fn: Callable[[str, str], bool]

    def check(self, y: Any, claim_text: str) -> bool:
        if not self.base.check(y, claim_text):
            return False
        y_str = y if isinstance(y, str) else str(y)
        return bool(self.nli_fn(y_str, claim_text))


# ---------------------------------------------------------------------------
# Replay interface
# ---------------------------------------------------------------------------

class Replayer(Protocol):
    """Deterministically replay one pipeline step."""

    def run(self, step: ReplayableStep, graph: GraphLike) -> bytes:
        ...


@dataclass
class CompositeReplayer:
    """Dispatch replayable operations to deterministic handlers."""

    handlers: dict[str, Callable[[ReplayableStep, GraphLike], bytes]] = field(default_factory=dict)

    def register(self, op: str, fn: Callable[[ReplayableStep, GraphLike], bytes]) -> None:
        self.handlers[op] = fn

    def run(self, step: ReplayableStep, graph: GraphLike) -> bytes:
        if step.op not in self.handlers:
            raise KeyError(
                f"No replay handler for op={step.op!r}. "
                f"Registered handlers: {sorted(self.handlers)}"
            )
        return self.handlers[step.op](step, graph)


# ---------------------------------------------------------------------------
# Structured check result
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Output of the four-channel checker."""

    passed: bool = False

    # v4 theorem channels
    integrity_ok: bool = True      # V_H
    replay_ok: bool = True         # V_Pi
    execution_ok: bool = True      # V_Gamma
    entailment_ok: bool = True     # V_entail

    reasons: list[str] = field(default_factory=list)
    replay_output: bytes | None = None
    replay_output_digest: str | None = None

    @property
    def V_H(self) -> bool:
        return self.integrity_ok

    @property
    def V_Pi(self) -> bool:
        return self.replay_ok

    @property
    def V_Gamma(self) -> bool:
        return self.execution_ok

    @property
    def V_entail(self) -> bool:
        return self.entailment_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "integrity_ok": self.integrity_ok,
            "replay_ok": self.replay_ok,
            "execution_ok": self.execution_ok,
            "entailment_ok": self.entailment_ok,
            "V_H": self.V_H,
            "V_Pi": self.V_Pi,
            "V_Gamma": self.V_Gamma,
            "V_entail": self.V_entail,
            "reasons": list(self.reasons),
            "replay_output_digest": self.replay_output_digest,
        }


# ---------------------------------------------------------------------------
# Compatibility helpers for different ExecutionContract shapes
# ---------------------------------------------------------------------------

def _get_any(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _to_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {str(k) for k in value.keys()}
    try:
        return {str(v) for v in value}
    except TypeError:
        return {str(value)}


def _policy_ids_from_rules(value: Any) -> set[str]:
    if not value:
        return set()
    out: set[str] = set()
    if isinstance(value, dict):
        value = value.values()
    for item in value:
        if isinstance(item, dict):
            pid = item.get("policy_id") or item.get("id") or item.get("name")
            if pid:
                out.add(str(pid))
        else:
            out.add(str(item))
    return out


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

@dataclass
class Checker:
    """External deterministic verifier for unified grounding certificates."""

    entailment: EntailmentChecker
    replayer: Replayer

    def check(self, cert: GroundingCertificate, graph: GraphLike) -> CheckResult:
        """Return the v4 four-channel CheckResult."""

        result = CheckResult()

        claim_ok, _ = self._check_claim_side(cert.claim_cert, graph, result)
        execution_ok = self._check_execution_side(cert.exec_cert, graph, result)

        result.passed = bool(claim_ok and execution_ok)
        return result

    def _check_claim_side(
        self,
        cc: ClaimCertificate,
        graph: GraphLike,
        result: CheckResult,
    ) -> tuple[bool, bytes | None]:
        nodes = graph.nodes

        # V_H: evidence commitment integrity
        for evidence_id, expected_digest in zip(cc.evidence_ids, cc.evidence_digests):
            node = nodes.get(evidence_id)
            if node is None:
                result.integrity_ok = False
                result.reasons.append(f"evidence_node_missing:{evidence_id}")
                continue

            if not isinstance(node, TruthNode):
                result.integrity_ok = False
                result.reasons.append(
                    f"evidence_not_truth:{evidence_id}:{type(node).__name__}"
                )
                continue

            actual_digest = H(node.content_for_hash())
            if actual_digest != expected_digest:
                result.integrity_ok = False
                result.reasons.append(f"hash_mismatch:{evidence_id}")

        if not result.integrity_ok:
            return False, None

        # V_Pi: deterministic replay consistency
        try:
            last_output = b""
            for step in cc.pipeline:
                last_output = self.replayer.run(step, graph)

            replay_digest = H(last_output)
            result.replay_output = last_output
            result.replay_output_digest = replay_digest

            expected_replay_digest = getattr(cc, "replay_output_digest", None)
            if not expected_replay_digest:
                result.replay_ok = False
                result.reasons.append("replay_digest_missing")
                return False, last_output

            if replay_digest != expected_replay_digest:
                result.replay_ok = False
                result.reasons.append(
                    "replay_digest_mismatch:"
                    f"expected={expected_replay_digest},got={replay_digest}"
                )
                return False, last_output

        except Exception as exc:  # noqa: BLE001
            result.replay_ok = False
            result.reasons.append(f"replay_exception:{type(exc).__name__}:{exc}")
            return False, None

        # V_entail: deterministic entailment under rule system R
        claim_node = nodes.get(cc.claim_id)
        claim_text = (
            getattr(claim_node, "canonical", "")
            or getattr(claim_node, "raw", "")
            or ""
        )

        if not claim_text:
            result.entailment_ok = False
            result.reasons.append(f"claim_text_missing:{cc.claim_id}")
            return False, last_output

        replay_text = last_output.decode("utf-8", errors="replace")
        if not self.entailment.check(replay_text, claim_text):
            result.entailment_ok = False
            result.reasons.append(f"entailment_rejected:claim={claim_text[:100]!r}")
            return False, last_output

        return True, last_output

    def _check_execution_side(
        self,
        ec: ExecutionCertificate,
        graph: GraphLike,
        result: CheckResult,
    ) -> bool:
        contract = ec.contract
        nodes = graph.nodes

        # Read contract fields in a compatibility-safe way. This supports both
        # tool_allowlist-style and allowed_tools-style schemas.
        tool_allowlist = _to_set(
            _get_any(contract, "tool_allowlist", "allowed_tools", default=set())
        )
        blocked_tools = _to_set(
            _get_any(contract, "blocked_tools", default=set())
        )

        memory_access = _to_set(
            _get_any(contract, "memory_access", default=set())
        )
        memory_policy = _get_any(contract, "memory_policy", default={}) or {}

        allowed_delegations = _to_set(
            _get_any(contract, "allowed_delegations", default=set())
        )
        delegation_policy = _get_any(contract, "delegation_policy", default={}) or {}

        required_schema_ids = _to_set(
            _get_any(contract, "required_schema_ids", "schema_ids", default=set())
        )
        schemas = _get_any(contract, "schemas", default={}) or {}
        required_schema_ids |= _to_set(schemas)

        required_policy_ids = _to_set(
            _get_any(contract, "required_policy_ids", default=set())
        )
        required_policy_ids |= _policy_ids_from_rules(
            _get_any(contract, "policy_rules", default=[])
        )

        max_tool_calls = _get_any(contract, "max_tool_calls", default=None)

        # V_Gamma(a): tool/function/MCP allow-list and block-list
        tool_call_count = 0
        for tool_id in getattr(ec, "tool_call_ids", ()):
            node = nodes.get(tool_id)

            if not isinstance(node, ToolCallNode):
                result.execution_ok = False
                result.reasons.append(f"tool_node_missing_or_wrong_type:{tool_id}")
                continue

            tool_call_count += 1
            tool_name = node.tool_name

            if blocked_tools and tool_name in blocked_tools:
                result.execution_ok = False
                result.reasons.append(f"tool_blocked:{tool_name}")

            if tool_allowlist and tool_name not in tool_allowlist:
                result.execution_ok = False
                result.reasons.append(f"tool_not_allowed:{tool_name}")

        if max_tool_calls is not None and tool_call_count > int(max_tool_calls):
            result.execution_ok = False
            result.reasons.append(f"tool_budget_exceeded:{tool_call_count}>{max_tool_calls}")

        # V_Gamma(b): memory access policy
        for memory_id in getattr(ec, "memory_node_ids", ()):
            node = nodes.get(memory_id)

            if not isinstance(node, MemoryNode):
                result.execution_ok = False
                result.reasons.append(f"memory_node_missing_or_wrong_type:{memory_id}")
                continue

            key = f"{node.scope}:{node.op}"

            if memory_access and key not in memory_access:
                result.execution_ok = False
                result.reasons.append(f"memory_access_violation:{key}")

            if isinstance(memory_policy, dict):
                if node.op == "read" and memory_policy.get("allow_reads") is False:
                    result.execution_ok = False
                    result.reasons.append(f"memory_read_blocked:{key}")

                if node.op == "write" and memory_policy.get("allow_writes") is False:
                    result.execution_ok = False
                    result.reasons.append(f"memory_write_blocked:{key}")

                allowed = memory_policy.get("allowed")
                if allowed is not None and key not in _to_set(allowed):
                    result.execution_ok = False
                    result.reasons.append(f"memory_access_not_in_policy:{key}")

        # V_Gamma(c): delegation policy
        allowed_agents_from_policy = set()
        if isinstance(delegation_policy, dict):
            allowed_agents_from_policy = _to_set(
                delegation_policy.get("allowed_agents")
                or delegation_policy.get("allowed_delegations")
            )

        effective_allowed_delegations = allowed_delegations | allowed_agents_from_policy

        for delegation_id in getattr(ec, "delegation_ids", ()):
            node = nodes.get(delegation_id)

            if not isinstance(node, DelegationEdge):
                result.execution_ok = False
                result.reasons.append(
                    f"delegation_node_missing_or_wrong_type:{delegation_id}"
                )
                continue

            if effective_allowed_delegations and node.child_agent not in effective_allowed_delegations:
                result.execution_ok = False
                result.reasons.append(f"delegation_not_allowed:{node.child_agent}")

        # V_Gamma(d): required schemas
        attached_schema_ids = {
            node.schema_id
            for sid in getattr(ec, "schema_node_ids", ())
            if isinstance((node := nodes.get(sid)), SchemaNode)
        }

        missing_schemas = required_schema_ids - attached_schema_ids
        if missing_schemas:
            result.execution_ok = False
            result.reasons.append(f"schema_missing:{sorted(missing_schemas)}")

        # V_Gamma(e): required policies / guardrails
        attached_policy_ids = {
            node.policy_id
            for pid in getattr(ec, "policy_node_ids", ())
            if isinstance((node := nodes.get(pid)), PolicyNode)
        }

        missing_policies = required_policy_ids - attached_policy_ids
        if missing_policies:
            result.execution_ok = False
            result.reasons.append(f"policy_missing:{sorted(missing_policies)}")

        return result.execution_ok


# ---------------------------------------------------------------------------
# Default replay handlers used by smoke tests and light experiments
# ---------------------------------------------------------------------------

def build_default_replayer() -> CompositeReplayer:
    """Return deterministic replay handlers for minimal pipelines."""

    replayer = CompositeReplayer()

    def _identity(step: ReplayableStep, graph: GraphLike) -> bytes:
        if not step.input_ids:
            return b""
        first = graph.nodes.get(step.input_ids[0])
        if isinstance(first, TruthNode):
            return first.payload
        return b""

    def _concat(step: ReplayableStep, graph: GraphLike) -> bytes:
        delimiter = str(step.params.get("delim", "\n")).encode("utf-8")
        chunks: list[bytes] = []
        for node_id in step.input_ids:
            node = graph.nodes.get(node_id)
            if isinstance(node, TruthNode):
                chunks.append(node.payload)
        return delimiter.join(chunks)

    replayer.register("identity", _identity)
    replayer.register("concat", _concat)

    return replayer