"""
Backend schema for the PCG-MAS interactive demo.

These dataclasses define the typed contracts between:
    pipeline.py        (produces these objects)
    server.py          (serializes them over SSE / JSON to the browser)
    static/app.js      (consumes & renders them)

They mirror the three appendix templates of the paper:
    Template 1  →  EvidenceItem, ToolOutput, AtomicClaim
    Template 2  →  ChannelVerdict, ClaimCertificate
    Template 3  →  RiskDecision, ResponsibilityReport

and they align with the existing pcg/ package signatures:
    pcg.certificate.GroundingCertificate
    pcg.checker.Checker / CheckResult
    pcg.responsibility.ResponsibilityEstimator
    pcg.risk.Action / ThresholdPolicy / posterior_risk
    pcg.eval.bootstrap.compare

Nothing in this file does any computation. It only defines the shapes.

NEVER add `meta` fields that leak local paths, usernames, or personal info.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Any


# =============================================================================
# 1. Evidence / tool-output / claim primitives (Template 1)
# =============================================================================

@dataclass
class EvidenceItem:
    """One retrieved evidence chunk, with content commitment.

    `hash` is sha256 of the normalized text. The commitment lets V_I (Integrity)
    detect tampering between certificate construction and replay.
    """
    id: str                              # "e1", "e2", ...
    text: str
    source: str                          # URL, filename, or "user_provided"
    hash: str
    span: Optional[tuple[int, int]] = None  # char offsets in source


@dataclass
class ToolOutput:
    """One tool invocation output, with commitment over args + output."""
    id: str                              # "t1", "t2", ...
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    hash: str = ""


@dataclass
class AtomicClaim:
    """One atomic claim extracted from the answer (Template 1).

    Each claim must cite at least one EvidenceItem.id or ToolOutput.id;
    claims with no citations are auto-rejected by V_Cov.
    """
    claim_id: str                        # "c1", "c2", ...
    claim_text: str
    support_ids: list[str] = field(default_factory=list)        # → EvidenceItem.id
    tool_output_ids: list[str] = field(default_factory=list)    # → ToolOutput.id
    confidence: float = 0.0                                     # in [0, 1]
    uncertainty_flags: list[str] = field(default_factory=list)
    # Allowed flags (per appendix):
    #   "missing_evidence" | "numeric_ambiguity" | "tool_dependency" | "none"


# =============================================================================
# 2. Five-channel verdicts (Template 2)
# =============================================================================

class ChannelName(str, Enum):
    """The 5 audit channels. Symbols are the user-facing canon.

    Footnote shown in the app:
        V_I    = Integrity      / evidence-commitment check
        V_R    = Replay         / support-pipeline replay
        V_D    = Drift          / semantic replay drift
        V_Ch   = Checker        / entailment + execution-contract check
        V_Cov  = Coverage       / whether cited support covers the claim
    """
    V_I   = "V_I"
    V_R   = "V_R"
    V_D   = "V_D"
    V_Ch  = "V_Ch"
    V_Cov = "V_Cov"


CHANNEL_LABELS: dict[ChannelName, str] = {
    ChannelName.V_I:   "Integrity",
    ChannelName.V_R:   "Replay",
    ChannelName.V_D:   "Drift",
    ChannelName.V_Ch:  "Checker",
    ChannelName.V_Cov: "Coverage",
}


class ChannelState(str, Enum):
    IDLE    = "idle"
    PENDING = "pending"
    PASS    = "pass"
    FAIL    = "fail"
    SKIP    = "skip"


@dataclass
class ChannelVerdict:
    """One channel's verdict on one claim."""
    channel: ChannelName
    state: ChannelState
    score: Optional[float] = None        # channel-specific score in [0, 1]
    detail: str = ""                     # short human-readable rationale
    elapsed_ms: int = 0


# =============================================================================
# 3. Per-claim certificate (Template 2)
# =============================================================================

@dataclass
class ClaimCertificate:
    """Audit envelope for one claim.

    Accepted iff all 5 channels pass. `integrity_hash` is the commitment over
    (claim_text, support_set, pipeline_id, contract_id) — re-derivable by V_I.
    """
    claim: AtomicClaim
    channels: dict[ChannelName, ChannelVerdict] = field(default_factory=dict)
    accepted: bool = False
    integrity_hash: str = ""
    minimal_support_ids: list[str] = field(default_factory=list)

    def failed_channels(self) -> list[ChannelName]:
        return [c for c, v in self.channels.items() if v.state == ChannelState.FAIL]


# =============================================================================
# 4. Mask-and-replay responsibility (Template 3 partial, pcg.responsibility API)
# =============================================================================

class ComponentType(str, Enum):
    EVIDENCE   = "evidence"
    TOOL       = "tool"
    SCHEMA     = "schema"
    MEMORY     = "memory"
    POLICY     = "policy"
    DELEGATION = "delegation"


@dataclass
class ResponsibilityScore:
    """One component's interventional responsibility estimate.

    Hooks into pcg.responsibility.ResponsibilityEstimator:
        - score      = MC estimate of Eq. 7 (Resp(e; Z, G_t))
        - ci_low/high = Hoeffding (Eq. 27) or Normal CI from M replays
    """
    component_id: str                    # e.g. "evidence:e2" or "tool:t1"
    component_type: ComponentType
    score: float                         # in [0, 1]
    ci_low: float = 0.0
    ci_high: float = 1.0
    rank: int = 0                        # 1 = most responsible


@dataclass
class ResponsibilityReport:
    """Mask-and-replay output for one claim."""
    claim_id: str
    scores: list[ResponsibilityScore] = field(default_factory=list)
    top_responsible_id: str = ""         # rank-1 component
    n_replays: int = 0                   # M from Theorem 3(i)
    rank_recovery_prob: float = 0.0      # Eq. 28
    ci_method: str = "hoeffding"         # "hoeffding" | "normal"


# =============================================================================
# 5. Audit-channel probe envelopes (paired bootstrap, pcg.eval.bootstrap API)
# =============================================================================

@dataclass
class AuditEnvelope:
    """Bootstrap CI on one channel's pass-rate across N samples in this run."""
    channel: ChannelName
    pass_rate: float
    ci_low: float
    ci_high: float
    n_samples: int                       # claims in this run
    n_bootstrap: int = 2000
    method: str = "paired_bootstrap"     # matches pcg.eval.bootstrap


# =============================================================================
# 6. Risk-aware control (Template 3, pcg.risk API)
# =============================================================================

class RiskAction(str, Enum):
    """The 4 actions of the risk controller. Mirrors pcg.risk.Action."""
    ANSWER   = "Answer"
    VERIFY   = "Verify"
    ESCALATE = "Escalate"
    REFUSE   = "Refuse"


@dataclass
class RiskDecision:
    """Final controller output.

    Hooks into pcg.risk.posterior_risk + pcg.risk.ThresholdPolicy:
        - posterior_risk = r(b, Z) from Eq. 24
        - expected_cost  = C(b, a) per action from Eq. 22
    """
    action: RiskAction
    posterior_risk: float                # in [0, 1]
    expected_cost: dict[RiskAction, float] = field(default_factory=dict)
    residual_risk: dict[RiskAction, float] = field(default_factory=dict)
    dominant_failure_channel: Optional[ChannelName] = None
    reason_codes: list[str] = field(default_factory=list)
    summary: str = ""


# =============================================================================
# 7. Top-level certificate
# =============================================================================

@dataclass
class RunMeta:
    """Metadata for one /api/run. No personal info, no local paths."""
    cert_id: str                         # ULID-like, generated server-side
    backend_label: str                   # "OpenAI · gpt-4o-mini"
    backend_provider: str                # "openai" | "anthropic" | ...
    backend_model: str                   # the model_id string
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    pipeline_id: str = "pcg_mas_v1"
    contract_id: str = "gamma_v1"
    started_at_iso: str = ""
    elapsed_ms_total: int = 0
    tokens_total: int = 0


@dataclass
class FullCertificate:
    """The top-level envelope returned to the browser.

    One FullCertificate per /api/run. Streamed incrementally over SSE so the
    frontend can render claims / channels / risk action as they materialize.
    """
    question: str
    answer_draft: str                    # before any rejection
    answer_final: str                    # may be withheld if risk action != ANSWER
    accepted: bool                       # convenience: action == ANSWER
    integrity_hash: str                  # top-level commitment

    evidence: list[EvidenceItem]                = field(default_factory=list)
    tool_outputs: list[ToolOutput]              = field(default_factory=list)
    claims: list[AtomicClaim]                   = field(default_factory=list)

    claim_certificates: list[ClaimCertificate]  = field(default_factory=list)
    responsibility:    list[ResponsibilityReport] = field(default_factory=list)
    audit_envelopes:   list[AuditEnvelope]      = field(default_factory=list)
    risk:              Optional[RiskDecision]   = None

    meta: Optional[RunMeta] = None


# =============================================================================
# 8. SSE event payloads (server → browser)
# =============================================================================

class SSEEventType(str, Enum):
    """Names match server.py's `event:` lines and app.js's handler keys."""
    START               = "start"
    EVIDENCE            = "evidence"            # evidence retrieval done
    CLAIM               = "claim"               # one claim extracted
    CHANNEL             = "channel"             # one channel verdict for one claim
    CLAIM_CERT          = "claim_cert"          # full ClaimCertificate ready
    RESPONSIBILITY      = "responsibility"      # one ResponsibilityReport ready
    AUDIT_ENVELOPE      = "audit_envelope"      # one AuditEnvelope ready
    RISK                = "risk"                # final RiskDecision
    CERTIFICATE         = "certificate"         # final FullCertificate
    ERROR               = "error"
    DONE                = "done"


@dataclass
class ChannelStreamEvent:
    """Compact SSE payload for one V_x update on one claim."""
    claim_id: str
    channel: ChannelName
    state: ChannelState
    score: Optional[float] = None
    detail: str = ""


# =============================================================================
# 9. Serialization helpers
# =============================================================================

def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses / enums / nested structures to JSON-safe dicts.

    Used by server.py to serialize Full / Claim / Responsibility certificates
    over SSE. Enums become their .value strings; tuples become lists.
    """
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    return str(obj)


__all__ = [
    # primitives
    "EvidenceItem", "ToolOutput", "AtomicClaim",
    # channels
    "ChannelName", "ChannelState", "ChannelVerdict", "CHANNEL_LABELS",
    # certificates
    "ClaimCertificate", "FullCertificate", "RunMeta",
    # responsibility
    "ComponentType", "ResponsibilityScore", "ResponsibilityReport",
    # envelopes
    "AuditEnvelope",
    # risk
    "RiskAction", "RiskDecision",
    # SSE
    "SSEEventType", "ChannelStreamEvent",
    # helpers
    "to_jsonable",
]
