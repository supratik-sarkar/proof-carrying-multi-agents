"""
Certificates (Definitions 2.4, 2.5, 2.7, 2.8 of the paper).

    - ReplayableStep         <--> phi_j in Definition 2.3 (pipeline)
    - ClaimCertificate       <--> Z^clm = (c, S, Pi, p, meta), Def 2.4, Eq. 4
    - ExecutionContract      <--> Gamma, Def 2.6
    - ExecutionCertificate   <--> Z^exe = (Pi, Gamma, meta), Def 2.7
    - GroundingCertificate   <--> Z = (c, S, Pi, Gamma, p, meta), Def 2.8

The key invariant is that *every* field in a certificate is either a hex digest
(referring to a node in G_t) or a plain dataclass — there are no Python object
references to live graph nodes. This is what makes certificates serializable
and independently re-verifiable: the Verifier receives a certificate and a
copy of G_t, and can re-check everything without access to the Prover's process.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pcg.commitments import H

# -----------------------------------------------------------------------------
# Pipeline steps  (Definition 2.3, Eq. 3)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayableStep:
    """A single step phi_j in the replayable pipeline Pi.

    Each step names a concrete operation (retrieval, parsing, alignment,
    schema validation, memory access, tool call, formatting) and declares the
    parameters needed to reproduce it deterministically. See Appendix A.2 of
    the paper for the replay contract.

    Fields:
        op: canonical operation name (e.g., "bm25_retrieve", "dpr_retrieve",
            "span_extract", "nli_entail", "schema_validate", "memory_read").
        version: implementation version (library version or git sha of the op).
        params: JSON-serializable parameter dict (top-k, thresholds, prompts,
            temperature, seed, ...).
        input_ids: list of node IDs consumed as input to this step.
        output_digest: hash of the output, to tie the step to a downstream
            TruthNode (if any).
    """

    op: str
    version: str
    params: dict[str, Any] = field(default_factory=dict)
    input_ids: tuple[str, ...] = ()
    output_digest: str | None = None

    def canonical_bytes(self) -> bytes:
        """Deterministic byte encoding used for pipeline-level digest."""
        return json.dumps(
            {
                "op": self.op,
                "v": self.version,
                "p": self.params,
                "i": list(self.input_ids),
                "o": self.output_digest or "",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


# -----------------------------------------------------------------------------
# Claim-side certificate (Definition 2.4, Eq. 4)
# -----------------------------------------------------------------------------


@dataclass
class ClaimCertificate:
    """Z^clm = (c, S, Pi, p, meta) from Definition 2.4.

    Attributes:
        claim_id: ID of the ClaimNode in G_t.
        evidence_ids: S, the evidence set. Each element is a TruthNode ID.
        evidence_digests: recorded h(v) for each v in S, in the same order as
            evidence_ids. The Checker will re-compute H(x(v)) and compare.
        pipeline: Pi, the replayable transformation chain.
        confidence: p in [0, 1], the declared confidence.
        replay_output_digest: hash of the replay output y = Pi(...); the checker
            verifies that re-running Pi produces the same y.
        meta: versioning, seeds, canonicalization flags (Appendix A.2).
    """

    claim_id: str
    evidence_ids: tuple[str, ...]
    evidence_digests: tuple[str, ...]
    pipeline: tuple[ReplayableStep, ...]
    confidence: float
    replay_output_digest: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if len(self.evidence_ids) != len(self.evidence_digests):
            raise ValueError("evidence_ids and evidence_digests must have equal length")

    def canonical_bytes(self) -> bytes:
        """For use as an atomic hash when the claim certificate is embedded
        inside a unified certificate."""
        return json.dumps(
            {
                "c": self.claim_id,
                "S": list(self.evidence_ids),
                "h_S": list(self.evidence_digests),
                "pi": [s.canonical_bytes().decode("utf-8") for s in self.pipeline],
                "p": round(self.confidence, 6),   # 6-digit quantization for stability
                "y": self.replay_output_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def digest(self) -> str:
        return H(self.canonical_bytes())


# -----------------------------------------------------------------------------
# Execution-side certificate (Definitions 2.6, 2.7, Eqs. 5a/5b)
# -----------------------------------------------------------------------------


@dataclass
class ExecutionContract:
    """Gamma from Definition 2.6: the finite set of execution constraints.

    This is the single most important object for answering the ICML referee's
    "why is this multi-agent?" question. Gamma is what distinguishes PCG from
    generic retrieval-grounded generation.

    v4 formulation (enhanced control):
        - allowed_tools:       set of explicitly permitted tool names/classes
        - blocked_tools:       set of explicitly forbidden tool names (takes precedence)
        - schemas:             dict mapping schema_id -> schema definition (JSON Schema)
        - memory_policy:       dict specifying which memory scopes may be read/written
        - delegation_policy:   dict specifying sub-agent invocation constraints
        - policy_rules:        list of policy/guardrail rules (structured clauses)
        - mcp_endpoints:       dict mapping MCP server names -> endpoint configs
        - resource_budget:     dict with max_tokens, max_latency_ms, max_tool_calls

    Backward compatibility: v2 contracts without new fields load with permissive defaults.
    """

    # v4 Enhanced tool control
    allowed_tools: frozenset[str] = frozenset()
    blocked_tools: frozenset[str] = frozenset()
    
    # v4 Schema registry (maps schema_id -> JSON Schema dict)
    schemas: dict[str, Any] = field(default_factory=dict)
    
    # v4 Refined memory policy (per-scope read/write constraints)
    memory_policy: dict[str, Any] = field(default_factory=dict)
    
    # v4 Refined delegation policy (sub-agent constraints)
    delegation_policy: dict[str, Any] = field(default_factory=dict)
    
    # v4 Policy/guardrail rules (structured clauses)
    policy_rules: list[dict[str, Any]] = field(default_factory=list)
    
    # v4 MCP endpoint configurations
    mcp_endpoints: dict[str, Any] = field(default_factory=dict)
    
    # Resource budget (unified v4)
    resource_budget: dict[str, int | float | None] = field(default_factory=lambda: {
        "max_tokens": None,
        "max_latency_ms": None,
        "max_tool_calls": None,
    })

    # --- v2 backward-compatibility fields (deprecated but kept for loading old certs) ---
    # These will be merged into v4 fields during load
    tool_allowlist: frozenset[str] = frozenset()
    memory_access: frozenset[str] = frozenset()
    allowed_delegations: frozenset[str] = frozenset()
    required_schema_ids: frozenset[str] = frozenset()
    required_policy_ids: frozenset[str] = frozenset()
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_latency_ms: float | None = None

    def __post_init__(self) -> None:
        """Normalize and validate: migrate v2 fields to v4 if needed."""
        # Ensure resource_budget dict has required keys
        if not self.resource_budget:
            self.resource_budget = {
                "max_tokens": None,
                "max_latency_ms": None,
                "max_tool_calls": None,
            }
        else:
            # Merge legacy max_* fields into resource_budget if present
            if self.max_tokens is not None:
                self.resource_budget.setdefault("max_tokens", self.max_tokens)
            if self.max_latency_ms is not None:
                self.resource_budget.setdefault("max_latency_ms", self.max_latency_ms)
            if self.max_tool_calls is not None:
                self.resource_budget.setdefault("max_tool_calls", self.max_tool_calls)
        
        # Migrate v2 tool_allowlist -> allowed_tools if needed
        if self.tool_allowlist and not self.allowed_tools:
            object.__setattr__(self, "allowed_tools", self.tool_allowlist)
        
        # Migrate v2 memory_access -> memory_policy if needed
        if self.memory_access and not self.memory_policy:
            self.memory_policy = {
                "allowed_scopes": sorted(self.memory_access)
            }
        
        # Migrate v2 allowed_delegations -> delegation_policy if needed
        if self.allowed_delegations and not self.delegation_policy:
            self.delegation_policy = {
                "allowed_delegations": sorted(self.allowed_delegations)
            }
        
        # Migrate v2 required_schema_ids into schemas if needed
        if self.required_schema_ids and not self.schemas:
            schema_dict = {}
            for sid in self.required_schema_ids:
                schema_dict[sid] = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}
            self.schemas = schema_dict
        
        # Migrate v2 required_policy_ids -> policy_rules if needed
        if self.required_policy_ids and not self.policy_rules:
            rules = []
            for pid in self.required_policy_ids:
                rules.append({"policy_id": pid, "kind": "guardrail"})
            self.policy_rules = rules

    def canonical_bytes(self) -> bytes:
        """Deterministic encoding using v4 fields; backward compatible."""
        return json.dumps(
            {
                "v": 4,  # v4 formulation marker
                "allowed_tools": sorted(self.allowed_tools),
                "blocked_tools": sorted(self.blocked_tools),
                "schemas": self.schemas,
                "memory_policy": self.memory_policy,
                "delegation_policy": self.delegation_policy,
                "policy_rules": sorted([json.dumps(r, sort_keys=True) for r in self.policy_rules]),
                "mcp_endpoints": self.mcp_endpoints,
                "resource_budget": {
                    "max_tokens": self.resource_budget.get("max_tokens"),
                    "max_latency_ms": self.resource_budget.get("max_latency_ms"),
                    "max_tool_calls": self.resource_budget.get("max_tool_calls"),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def digest(self) -> str:
        return H(self.canonical_bytes())

    # ---- v4 helper methods (for accessing v2 fields via v4 names) ----
    def get_allowed_tools(self) -> frozenset[str]:
        """Get allowed tools from v4 field, fall back to v2 field if needed."""
        return self.allowed_tools or self.tool_allowlist
    
    def get_memory_scopes(self) -> frozenset[str]:
        """Get memory access scopes from v4 field, fall back to v2 field."""
        if self.memory_policy and "allowed_scopes" in self.memory_policy:
            return frozenset(self.memory_policy["allowed_scopes"])
        return self.memory_access
    
    def get_allowed_delegations(self) -> frozenset[str]:
        """Get allowed delegations from v4 field, fall back to v2 field."""
        if self.delegation_policy and "allowed_delegations" in self.delegation_policy:
            return frozenset(self.delegation_policy["allowed_delegations"])
        return self.allowed_delegations
    
    def get_required_schema_ids(self) -> frozenset[str]:
        """Get required schema IDs from v4 field, fall back to v2 field."""
        if self.schemas:
            return frozenset(self.schemas.keys())
        return self.required_schema_ids
    
    def get_required_policy_ids(self) -> frozenset[str]:
        """Get required policy IDs from v4 field, fall back to v2 field."""
        if self.policy_rules:
            return frozenset(r.get("policy_id") for r in self.policy_rules if "policy_id" in r)
        return self.required_policy_ids
    
    @classmethod
    def from_v2(cls,
                tool_allowlist: frozenset[str] = frozenset(),
                memory_access: frozenset[str] = frozenset(),
                allowed_delegations: frozenset[str] = frozenset(),
                required_schema_ids: frozenset[str] = frozenset(),
                required_policy_ids: frozenset[str] = frozenset(),
                max_tool_calls: int | None = None,
                max_tokens: int | None = None,
                max_latency_ms: float | None = None) -> "ExecutionContract":
        """Factory method: create a v4 contract from v2 fields (backward compat)."""
        return cls(
            allowed_tools=tool_allowlist,
            memory_policy={"allowed_scopes": sorted(memory_access)} if memory_access else {},
            delegation_policy={"allowed_delegations": sorted(allowed_delegations)} if allowed_delegations else {},
            schemas={sid: {} for sid in required_schema_ids},
            policy_rules=[{"policy_id": pid, "kind": "guardrail"} for pid in required_policy_ids],
            resource_budget={
                "max_tokens": max_tokens,
                "max_latency_ms": max_latency_ms,
                "max_tool_calls": max_tool_calls,
            },
            # Preserve v2 fields for compatibility
            tool_allowlist=tool_allowlist,
            memory_access=memory_access,
            allowed_delegations=allowed_delegations,
            required_schema_ids=required_schema_ids,
            required_policy_ids=required_policy_ids,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            max_latency_ms=max_latency_ms,
        )


@dataclass
class ExecutionCertificate:
    """Z^exe = (Pi, Gamma, meta) from Definition 2.7."""

    pipeline: tuple[ReplayableStep, ...]
    contract: ExecutionContract
    # IDs of the logged ToolCall / Memory / Delegation nodes that were
    # actually used during the run. The checker compares these against
    # the contract.
    tool_call_ids: tuple[str, ...] = ()
    memory_node_ids: tuple[str, ...] = ()
    delegation_ids: tuple[str, ...] = ()
    schema_node_ids: tuple[str, ...] = ()
    policy_node_ids: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "pi": [s.canonical_bytes().decode("utf-8") for s in self.pipeline],
                "Gamma": self.contract.digest(),
                "tc": list(self.tool_call_ids),
                "mem": list(self.memory_node_ids),
                "del": list(self.delegation_ids),
                "sch": list(self.schema_node_ids),
                "pol": list(self.policy_node_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def digest(self) -> str:
        return H(self.canonical_bytes())


# -----------------------------------------------------------------------------
# Unified grounding certificate (Definition 2.8, Eq. 6)
# -----------------------------------------------------------------------------


@dataclass
class GroundingCertificate:
    """Z = (c, S, Pi, Gamma, p, meta) from Definition 2.8.

    This is THE artifact that gets serialized, transmitted, and checked.
    It is designed to be round-trippable through JSON: `to_json()` and
    `from_json()` are loss-free modulo float precision.
    """

    claim_cert: ClaimCertificate
    exec_cert: ExecutionCertificate
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def claim_id(self) -> str:
        return self.claim_cert.claim_id

    @property
    def confidence(self) -> float:
        return self.claim_cert.confidence

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.claim_cert.evidence_ids

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "clm": self.claim_cert.digest(),
                "exe": self.exec_cert.digest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def digest(self) -> str:
        """A single hex digest uniquely identifying this certificate.

        Used for deduplication, caching, and as the certificate ID in
        downstream logs.
        """
        return H(self.canonical_bytes())

    # ---- serialization (v4 with backward compat) ----
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict using v4 format (includes backward-compat hints)."""
        contract = self.exec_cert.contract
        return {
            "claim_cert": {
                "claim_id": self.claim_cert.claim_id,
                "evidence_ids": list(self.claim_cert.evidence_ids),
                "evidence_digests": list(self.claim_cert.evidence_digests),
                "pipeline": [
                    {"op": s.op, "version": s.version, "params": s.params,
                     "input_ids": list(s.input_ids), "output_digest": s.output_digest}
                    for s in self.claim_cert.pipeline
                ],
                "confidence": self.claim_cert.confidence,
                "replay_output_digest": self.claim_cert.replay_output_digest,
                "meta": self.claim_cert.meta,
            },
            "exec_cert": {
                "pipeline": [
                    {"op": s.op, "version": s.version, "params": s.params,
                     "input_ids": list(s.input_ids), "output_digest": s.output_digest}
                    for s in self.exec_cert.pipeline
                ],
                "contract": {
                    # v4 fields
                    "version": 4,
                    "allowed_tools": sorted(contract.allowed_tools),
                    "blocked_tools": sorted(contract.blocked_tools),
                    "schemas": contract.schemas,
                    "memory_policy": contract.memory_policy,
                    "delegation_policy": contract.delegation_policy,
                    "policy_rules": contract.policy_rules,
                    "mcp_endpoints": contract.mcp_endpoints,
                    "resource_budget": contract.resource_budget,
                    # v2 backward-compat fields (for old certificate readers)
                    "tool_allowlist": sorted(contract.tool_allowlist),
                    "memory_access": sorted(contract.memory_access),
                    "allowed_delegations": sorted(contract.allowed_delegations),
                    "required_schema_ids": sorted(contract.required_schema_ids),
                    "required_policy_ids": sorted(contract.required_policy_ids),
                    "max_tool_calls": contract.max_tool_calls,
                    "max_tokens": contract.max_tokens,
                    "max_latency_ms": contract.max_latency_ms,
                },
                "tool_call_ids": list(self.exec_cert.tool_call_ids),
                "memory_node_ids": list(self.exec_cert.memory_node_ids),
                "delegation_ids": list(self.exec_cert.delegation_ids),
                "schema_node_ids": list(self.exec_cert.schema_node_ids),
                "policy_node_ids": list(self.exec_cert.policy_node_ids),
                "meta": self.exec_cert.meta,
            },
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GroundingCertificate":
        """Deserialize from dict with v4/v2 backward compatibility."""
        cc_d = d["claim_cert"]
        ec_d = d["exec_cert"]
        claim_cert = ClaimCertificate(
            claim_id=cc_d["claim_id"],
            evidence_ids=tuple(cc_d["evidence_ids"]),
            evidence_digests=tuple(cc_d["evidence_digests"]),
            pipeline=tuple(
                ReplayableStep(
                    op=s["op"], version=s["version"], params=s["params"],
                    input_ids=tuple(s["input_ids"]), output_digest=s["output_digest"],
                )
                for s in cc_d["pipeline"]
            ),
            confidence=float(cc_d["confidence"]),
            replay_output_digest=cc_d["replay_output_digest"],
            meta=cc_d.get("meta", {}),
        )
        
        # Load contract with v4/v2 compatibility
        contract_d = ec_d["contract"]
        contract_version = contract_d.get("version", 2)  # Default to v2 if not specified
        
        if contract_version >= 4:
            # v4 format
            contract = ExecutionContract(
                allowed_tools=frozenset(contract_d.get("allowed_tools", [])),
                blocked_tools=frozenset(contract_d.get("blocked_tools", [])),
                schemas=contract_d.get("schemas", {}),
                memory_policy=contract_d.get("memory_policy", {}),
                delegation_policy=contract_d.get("delegation_policy", {}),
                policy_rules=contract_d.get("policy_rules", []),
                mcp_endpoints=contract_d.get("mcp_endpoints", {}),
                resource_budget=contract_d.get("resource_budget", {
                    "max_tokens": None,
                    "max_latency_ms": None,
                    "max_tool_calls": None,
                }),
                # Preserve v2 fields for backward compat metadata
                tool_allowlist=frozenset(contract_d.get("tool_allowlist", [])),
                memory_access=frozenset(contract_d.get("memory_access", [])),
                allowed_delegations=frozenset(contract_d.get("allowed_delegations", [])),
                required_schema_ids=frozenset(contract_d.get("required_schema_ids", [])),
                required_policy_ids=frozenset(contract_d.get("required_policy_ids", [])),
                max_tool_calls=contract_d.get("max_tool_calls"),
                max_tokens=contract_d.get("max_tokens"),
                max_latency_ms=contract_d.get("max_latency_ms"),
            )
        else:
            # v2 format: upgrade to v4 with permissive defaults
            v2_tools = frozenset(contract_d.get("tool_allowlist", []))
            v2_memory = frozenset(contract_d.get("memory_access", []))
            v2_delegation = frozenset(contract_d.get("allowed_delegations", []))
            v2_schemas = frozenset(contract_d.get("required_schema_ids", []))
            v2_policies = frozenset(contract_d.get("required_policy_ids", []))
            
            contract = ExecutionContract(
                allowed_tools=v2_tools,
                blocked_tools=frozenset(),
                schemas={sid: {} for sid in v2_schemas},
                memory_policy={"allowed_scopes": sorted(v2_memory)} if v2_memory else {},
                delegation_policy={"allowed_delegations": sorted(v2_delegation)} if v2_delegation else {},
                policy_rules=[{"policy_id": pid, "kind": "guardrail"} for pid in v2_policies],
                mcp_endpoints={},
                resource_budget={
                    "max_tokens": contract_d.get("max_tokens"),
                    "max_latency_ms": contract_d.get("max_latency_ms"),
                    "max_tool_calls": contract_d.get("max_tool_calls"),
                },
                # Preserve v2 fields
                tool_allowlist=v2_tools,
                memory_access=v2_memory,
                allowed_delegations=v2_delegation,
                required_schema_ids=v2_schemas,
                required_policy_ids=v2_policies,
                max_tool_calls=contract_d.get("max_tool_calls"),
                max_tokens=contract_d.get("max_tokens"),
                max_latency_ms=contract_d.get("max_latency_ms"),
            )
        
        exec_cert = ExecutionCertificate(
            pipeline=tuple(
                ReplayableStep(
                    op=s["op"], version=s["version"], params=s["params"],
                    input_ids=tuple(s["input_ids"]), output_digest=s["output_digest"],
                )
                for s in ec_d["pipeline"]
            ),
            contract=contract,
            tool_call_ids=tuple(ec_d["tool_call_ids"]),
            memory_node_ids=tuple(ec_d["memory_node_ids"]),
            delegation_ids=tuple(ec_d["delegation_ids"]),
            schema_node_ids=tuple(ec_d["schema_node_ids"]),
            policy_node_ids=tuple(ec_d["policy_node_ids"]),
            meta=ec_d.get("meta", {}),
        )
        return cls(claim_cert=claim_cert, exec_cert=exec_cert, meta=d.get("meta", {}))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "GroundingCertificate":
        return cls.from_dict(json.loads(s))
