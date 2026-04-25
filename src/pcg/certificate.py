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
    generic retrieval-grounded generation: it covers

        - tool_allowlist: which tools may be called
        - memory_access:  which memory scopes may be read/written
        - delegation:     which sub-agents may be invoked and with what args
        - schema_ids:     required validator schemas for structured outputs
        - policy_ids:     guardrail clauses that must not be violated

    A unified certificate's execution side (Check_exe) verifies that the logged
    run metadata satisfies ALL of these.
    """

    tool_allowlist: frozenset[str] = frozenset()
    memory_access: frozenset[str] = frozenset()      # e.g. {"short_term:read", "persistent:read"}
    allowed_delegations: frozenset[str] = frozenset()
    required_schema_ids: frozenset[str] = frozenset()
    required_policy_ids: frozenset[str] = frozenset()
    # Optional resource budgets that are themselves contract clauses.
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_latency_ms: float | None = None

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "tool": sorted(self.tool_allowlist),
                "mem": sorted(self.memory_access),
                "del": sorted(self.allowed_delegations),
                "schema": sorted(self.required_schema_ids),
                "pol": sorted(self.required_policy_ids),
                "max_tc": self.max_tool_calls,
                "max_tok": self.max_tokens,
                "max_lat": self.max_latency_ms,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def digest(self) -> str:
        return H(self.canonical_bytes())


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

    # ---- serialization ----
    def to_dict(self) -> dict[str, Any]:
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
                    "tool_allowlist": sorted(self.exec_cert.contract.tool_allowlist),
                    "memory_access": sorted(self.exec_cert.contract.memory_access),
                    "allowed_delegations": sorted(self.exec_cert.contract.allowed_delegations),
                    "required_schema_ids": sorted(self.exec_cert.contract.required_schema_ids),
                    "required_policy_ids": sorted(self.exec_cert.contract.required_policy_ids),
                    "max_tool_calls": self.exec_cert.contract.max_tool_calls,
                    "max_tokens": self.exec_cert.contract.max_tokens,
                    "max_latency_ms": self.exec_cert.contract.max_latency_ms,
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
        contract = ExecutionContract(
            tool_allowlist=frozenset(ec_d["contract"]["tool_allowlist"]),
            memory_access=frozenset(ec_d["contract"]["memory_access"]),
            allowed_delegations=frozenset(ec_d["contract"]["allowed_delegations"]),
            required_schema_ids=frozenset(ec_d["contract"]["required_schema_ids"]),
            required_policy_ids=frozenset(ec_d["contract"]["required_policy_ids"]),
            max_tool_calls=ec_d["contract"]["max_tool_calls"],
            max_tokens=ec_d["contract"]["max_tokens"],
            max_latency_ms=ec_d["contract"]["max_latency_ms"],
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
