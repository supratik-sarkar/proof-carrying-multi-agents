"""
Demo-runtime master pipeline for PCG-MAS.

Orchestrates Phases 1-3 into a single SSE-streaming end-to-end run:

    START
      └─ evidence retrieval / parsing  -> EVIDENCE event
      └─ claim extraction (Phase 2a)   -> CLAIM events
      └─ 5-channel checker (Phase 2b)  -> CHANNEL events + CLAIM_CERT events
      └─ redundancy selection          -> (folded into RISK summary)
      └─ mask-and-replay responsibility -> RESPONSIBILITY events
      └─ audit envelopes               -> AUDIT_ENVELOPE events
      └─ risk-aware control            -> RISK event
      └─ final certificate             -> CERTIFICATE event
    DONE

Backward-compatible legacy entrypoint `run_pipeline(...)` is preserved so
server.py keeps working through Phase 5. It yields ChannelEvent + a single
FinalCertificate using the OLD 4-channel labels (V_H/V_pi/V_gamma/V_entail)
synthesized from the new 5-channel certificates (V_I/V_R/V_D/V_Ch/V_Cov).

The new entrypoint `run_pcg_mas(...)` yields the rich event stream.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Optional

from pcg_glue.backends import BackendChoice
from pcg_glue.sources import ResolvedInput, Evidence

# Phase 1 schemas
from pcg_glue.schemas import (
    EvidenceItem, ToolOutput, AtomicClaim,
    ChannelName, ChannelState, ChannelVerdict, CHANNEL_LABELS,
    ClaimCertificate, FullCertificate, RunMeta,
    ResponsibilityReport, AuditEnvelope,
    RiskDecision, RiskAction,
    SSEEventType, ChannelStreamEvent,
    to_jsonable,
)

# Phase 2 logic
from pcg_glue.claim_extractor import (
    extract_claims, commit_evidence, commit_tool_outputs,
)
from pcg_glue.channels import run_all_channels

# Phase 3 logic
from pcg_glue.responsibility   import estimate_responsibility_for_claim
from pcg_glue.risk_control     import decide as risk_decide
from pcg_glue.audit_envelopes  import compute_envelopes
from pcg_glue.redundancy       import select_independent


# =============================================================================
# Legacy event shape (kept for server.py backward-compat through Phase 5)
# =============================================================================

@dataclass
class ChannelEvent:
    """Legacy 4-channel event the current server.py expects. Synthesized from
    the new 5-channel certificates so the existing SSE consumer keeps working."""
    channel: str                    # "V_H" | "V_pi" | "V_gamma" | "V_entail"
    state: str                      # "pending" | "pass" | "fail"
    label: str
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class FinalCertificate:
    """Legacy top-level certificate the current server.py expects."""
    accepted: bool
    claim: str
    answer: str
    channels: dict[str, Any]
    backend: dict[str, Any]
    source: dict[str, Any]
    integrity_hash: str
    cert_id: str
    timestamp: float


# =============================================================================
# Adapter: ResolvedInput  ->  schema EvidenceItem[]
# =============================================================================

def _normalize_text(s: str) -> str:
    return " ".join((s or "").split())


def _adapt_evidence(resolved: ResolvedInput) -> tuple[list[EvidenceItem], str]:
    """Convert the existing ResolvedInput's evidence to schema EvidenceItem[]
    + return the question text."""
    out: list[EvidenceItem] = []
    for i, ev in enumerate(getattr(resolved, "evidence", []) or []):
        text = getattr(ev, "text", "") or ""
        # Source attribute could be .publisher (most common) or .source
        source = (
            getattr(ev, "publisher", None)
            or getattr(ev, "source", None)
            or getattr(ev, "url", None)
            or "unknown"
        )
        # Skip the synthetic "no evidence supplied" placeholder used by the
        # legacy pipeline when the user gives only a question.
        if isinstance(text, str) and text.startswith("(no evidence"):
            continue
        item = EvidenceItem(
            id=f"e{i + 1}",
            text=text,
            source=str(source),
            hash=hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest(),
        )
        out.append(item)
    question = getattr(resolved, "claim", "") or getattr(resolved, "question", "")
    return out, str(question)


def _cert_id(prefix: str, question: str) -> str:
    h = hashlib.sha256(f"{prefix}|{question}|{time.time()}".encode("utf-8")).hexdigest()[:16]
    return f"pcg-{prefix}-{h}"


# =============================================================================
# SSE event payload helper
# =============================================================================

def _evt(event_type: SSEEventType, payload: Any) -> dict[str, Any]:
    """Standardized SSE-yieldable dict: {event: "...", data: {...}}."""
    return {"event": event_type.value, "data": to_jsonable(payload)}


# =============================================================================
# NEW master entrypoint — yields rich SSE events
# =============================================================================

def run_pcg_mas(
    resolved: ResolvedInput,
    backend: BackendChoice,
    api_key: str,
    *,
    top_k_evidence: int = 6,
    do_responsibility: bool = True,
    do_envelopes: bool = True,
    do_redundancy: bool = True,
    responsibility_n_replays: int = 4,
    redundancy_k: int = 3,
) -> Iterator[dict[str, Any]]:
    """Full demo-runtime PCG-MAS pipeline. Yields SSE event dicts.

    Path-A execution (always run everything). To tighten latency/cost,
    callers can set do_responsibility / do_envelopes / do_redundancy to False.
    """
    t_total = time.time()
    cert_id = _cert_id("run", getattr(resolved, "claim", ""))

    # ---- Cap evidence at top_k for context window safety ----
    try:
        from .sources import top_k as topk_fn
        resolved = topk_fn(resolved, top_k_evidence)
    except Exception:
        pass

    evidence, question = _adapt_evidence(resolved)
    tools: list[ToolOutput] = []  # demo doesn't run live tools

    # Recommit hashes (ensures the schema's `hash` is the canonical sha256)
    evidence = commit_evidence(evidence)

    yield _evt(SSEEventType.START, {
        "cert_id": cert_id,
        "n_evidence": len(evidence),
        "backend": {"provider": backend.provider, "model": backend.model_id},
    })

    yield _evt(SSEEventType.EVIDENCE, {
        "items": [{"id": e.id, "source": e.source,
                   "text": e.text[:300], "hash": e.hash[:16]}
                  for e in evidence],
    })

    # =============================================================
    # Phase 2a: claim extraction
    # =============================================================
    try:
        answer_draft, claims, extract_meta = extract_claims(
            question=question,
            evidence=evidence,
            tool_outputs=tools,
            backend=backend,
            api_key=api_key,
        )
    except Exception as e:
        yield _evt(SSEEventType.ERROR, {
            "stage": "claim_extraction",
            "type": type(e).__name__,
            "message": str(e)[:300],
        })
        yield _evt(SSEEventType.DONE, {"elapsed_ms": int((time.time() - t_total) * 1000)})
        return

    for c in claims:
        yield _evt(SSEEventType.CLAIM, {
            "claim_id": c.claim_id,
            "claim_text": c.claim_text,
            "support_ids": c.support_ids,
            "tool_output_ids": c.tool_output_ids,
            "confidence": c.confidence,
            "uncertainty_flags": c.uncertainty_flags,
        })

    if not claims:
        # No claims extracted — produce a minimal certificate and stop.
        # The risk controller will see [] and choose Refuse.
        risk = risk_decide([])
        yield _evt(SSEEventType.RISK, risk)
        full = FullCertificate(
            question=question,
            answer_draft=answer_draft,
            answer_final=f"(no claims extracted: {extract_meta.get('abstain_reason', 'parser')})",
            accepted=False,
            integrity_hash=hashlib.sha256(cert_id.encode()).hexdigest(),
            evidence=evidence,
            claims=[],
            claim_certificates=[],
            responsibility=[],
            audit_envelopes=[],
            risk=risk,
            meta=RunMeta(
                cert_id=cert_id,
                backend_label=getattr(backend, "label", ""),
                backend_provider=backend.provider,
                backend_model=backend.model_id,
                elapsed_ms_total=int((time.time() - t_total) * 1000),
                tokens_total=extract_meta.get("tokens_out", 0),
            ),
        )
        yield _evt(SSEEventType.CERTIFICATE, full)
        yield _evt(SSEEventType.DONE, {"elapsed_ms": int((time.time() - t_total) * 1000)})
        return

    # =============================================================
    # Phase 2b: 5-channel checker per claim
    # =============================================================
    evidence_index = {e.id: e for e in evidence}
    tool_index = {t.id: t for t in tools}
    claim_certs: list[ClaimCertificate] = []

    for c in claims:
        # Emit pending state for all 5 channels so the UI lights them up
        for ch in [ChannelName.V_I, ChannelName.V_R, ChannelName.V_D,
                   ChannelName.V_Ch, ChannelName.V_Cov]:
            yield _evt(SSEEventType.CHANNEL, ChannelStreamEvent(
                claim_id=c.claim_id, channel=ch, state=ChannelState.PENDING,
            ))

        try:
            cc = run_all_channels(
                claim=c,
                evidence_index=evidence_index,
                tool_index=tool_index,
                question=question,
                backend=backend,
                api_key=api_key,
            )
        except Exception as e:
            # Synthesize a fully-FAIL ClaimCertificate so downstream still works
            cc = ClaimCertificate(
                claim=c, channels={
                    ch: ChannelVerdict(channel=ch, state=ChannelState.FAIL,
                                       score=0.0,
                                       detail=f"channel run crashed: {type(e).__name__}")
                    for ch in [ChannelName.V_I, ChannelName.V_R, ChannelName.V_D,
                               ChannelName.V_Ch, ChannelName.V_Cov]
                },
                accepted=False,
                integrity_hash="",
                minimal_support_ids=c.support_ids[:],
            )

        for ch_name, verdict in cc.channels.items():
            yield _evt(SSEEventType.CHANNEL, ChannelStreamEvent(
                claim_id=c.claim_id,
                channel=ch_name,
                state=verdict.state,
                score=verdict.score,
                detail=verdict.detail,
            ))

        yield _evt(SSEEventType.CLAIM_CERT, cc)
        claim_certs.append(cc)

    # =============================================================
    # Phase 3: redundancy / responsibility / envelopes / risk
    # =============================================================
    if do_redundancy:
        sel = select_independent(claim_certs, k=redundancy_k)
    else:
        sel = None

    responsibility_reports: list[ResponsibilityReport] = []
    if do_responsibility:
        # Only run on accepted claims with at least one cited support — masking
        # an already-failed or uncited claim is uninformative.
        for cc in claim_certs:
            if not cc.accepted:
                continue
            if not cc.claim.support_ids and not cc.claim.tool_output_ids:
                continue
            try:
                rep = estimate_responsibility_for_claim(
                    claim=cc.claim,
                    claim_cert=cc,
                    evidence=evidence,
                    tools=tools,
                    question=question,
                    backend=backend,
                    api_key=api_key,
                    n_replays=responsibility_n_replays,
                )
                responsibility_reports.append(rep)
                yield _evt(SSEEventType.RESPONSIBILITY, rep)
            except Exception as e:
                yield _evt(SSEEventType.ERROR, {
                    "stage": "responsibility",
                    "claim_id": cc.claim.claim_id,
                    "type": type(e).__name__,
                    "message": str(e)[:300],
                })

    envelopes: list[AuditEnvelope] = []
    if do_envelopes:
        envelopes = compute_envelopes(claim_certs)
        for env in envelopes:
            yield _evt(SSEEventType.AUDIT_ENVELOPE, env)

    risk = risk_decide(claim_certs)
    yield _evt(SSEEventType.RISK, risk)

    # =============================================================
    # Top-level FullCertificate
    # =============================================================
    answer_final = answer_draft if risk.action == RiskAction.ANSWER else \
        f"(answer withheld — controller chose {risk.action.value})"

    payload_for_hash = json.dumps({
        "question": question,
        "answer_draft": answer_draft,
        "claims": [c.claim_id for c in claims],
        "risk_action": risk.action.value,
    }, sort_keys=True)
    integrity_hash = hashlib.sha256(payload_for_hash.encode("utf-8")).hexdigest()

    full = FullCertificate(
        question=question,
        answer_draft=answer_draft,
        answer_final=answer_final,
        accepted=(risk.action == RiskAction.ANSWER),
        integrity_hash=integrity_hash,
        evidence=evidence,
        tool_outputs=tools,
        claims=claims,
        claim_certificates=claim_certs,
        responsibility=responsibility_reports,
        audit_envelopes=envelopes,
        risk=risk,
        meta=RunMeta(
            cert_id=cert_id,
            backend_label=getattr(backend, "label", ""),
            backend_provider=backend.provider,
            backend_model=backend.model_id,
            elapsed_ms_total=int((time.time() - t_total) * 1000),
            tokens_total=int(extract_meta.get("tokens_out", 0)),
            prompt_hashes={"claim_extractor": "v1", "checker": "v1"},
        ),
    )
    # Fold redundancy selection into the certificate's meta for inspection
    if sel is not None:
        full.meta.prompt_hashes["redundancy_selected"] = ",".join(sel.selected_ids)

    yield _evt(SSEEventType.CERTIFICATE, full)
    yield _evt(SSEEventType.DONE, {"elapsed_ms": int((time.time() - t_total) * 1000)})


# =============================================================================
# LEGACY entrypoint — kept for server.py backward-compat through Phase 5
# =============================================================================

# Map new 5-channel names to old 4-channel names used by the existing server SSE
_LEGACY_CHANNEL_MAP = {
    ChannelName.V_I:   "V_H",        # Integrity ~ proposer integrity
    ChannelName.V_R:   "V_pi",       # Replay
    ChannelName.V_D:   "V_pi",       # Drift folds into Replay channel
    ChannelName.V_Ch:  "V_gamma",    # Checker == judge
    ChannelName.V_Cov: "V_entail",   # Coverage ~ entailment in old vocab
}


def run_pipeline(
    resolved: ResolvedInput,
    backend: BackendChoice,
    api_key: str,
    *,
    replay_check: bool = True,
    top_k_evidence: int = 5,
) -> Iterator[Any]:
    """Backward-compatible entrypoint with the legacy event shape.

    Yields ChannelEvent | FinalCertificate, just like the old pipeline.
    Internally calls run_pcg_mas() and remaps the new SSE events.

    The server.py rewrite in Phase 5 will switch to run_pcg_mas() directly.
    """
    legacy_channel_state: dict[str, str] = {
        "V_H": "idle", "V_pi": "idle", "V_gamma": "idle", "V_entail": "idle",
    }

    final_certificate: Optional[FullCertificate] = None
    answer_text = ""
    question_text = getattr(resolved, "claim", "")

    for ev in run_pcg_mas(
        resolved, backend, api_key,
        top_k_evidence=top_k_evidence,
        do_responsibility=False,    # legacy mode: skip the heavy bits for parity
        do_envelopes=False,
        do_redundancy=False,
    ):
        ev_type = ev.get("event")
        data = ev.get("data", {})

        if ev_type == SSEEventType.CHANNEL.value:
            # data: {claim_id, channel, state, score, detail}
            new_ch = data.get("channel", "")
            new_state = data.get("state", "")
            try:
                legacy_ch = _LEGACY_CHANNEL_MAP[ChannelName(new_ch)]
            except (ValueError, KeyError):
                continue
            # Aggregate: legacy channel passes only if ALL contributing new
            # channels pass; fails if ANY fails; pending otherwise.
            if new_state == "fail":
                legacy_channel_state[legacy_ch] = "fail"
            elif new_state == "pass" and legacy_channel_state[legacy_ch] != "fail":
                legacy_channel_state[legacy_ch] = "pass"
            elif new_state == "pending" and legacy_channel_state[legacy_ch] == "idle":
                legacy_channel_state[legacy_ch] = "pending"
            yield ChannelEvent(
                channel=legacy_ch,
                state=legacy_channel_state[legacy_ch],
                label=new_ch,
                detail=data.get("detail", "")[:200],
            )

        elif ev_type == SSEEventType.CERTIFICATE.value:
            # The whole FullCertificate dict — store for final emit
            final_certificate = data    # type: ignore[assignment]
            answer_text = data.get("answer_final") or data.get("answer_draft") or ""

    # Emit the legacy FinalCertificate at the end
    if final_certificate is None:
        yield FinalCertificate(
            accepted=False, claim=question_text,
            answer="(no certificate produced)",
            channels={k: {"state": v} for k, v in legacy_channel_state.items()},
            backend={"provider": backend.provider, "model": backend.model_id},
            source={"kind": getattr(resolved, "source_kind", ""),
                    "label": getattr(resolved, "source_label", ""),
                    "n_evidence": len(getattr(resolved, "evidence", []) or [])},
            integrity_hash="",
            cert_id=_cert_id("legacy", question_text),
            timestamp=time.time(),
        )
        return

    fc_meta = final_certificate.get("meta", {}) or {}
    yield FinalCertificate(
        accepted=bool(final_certificate.get("accepted")),
        claim=question_text,
        answer=answer_text or "(no answer)",
        channels={
            "V_H":      {"state": legacy_channel_state["V_H"]},
            "V_pi":     {"state": legacy_channel_state["V_pi"]},
            "V_gamma":  {"state": legacy_channel_state["V_gamma"]},
            "V_entail": {"state": legacy_channel_state["V_entail"],
                         "rationale": (final_certificate.get("risk") or {}).get("summary", "")},
        },
        backend={"provider": backend.provider, "model": backend.model_id},
        source={
            "kind":  getattr(resolved, "source_kind",  ""),
            "label": getattr(resolved, "source_label", ""),
            "n_evidence": len(getattr(resolved, "evidence", []) or []),
        },
        integrity_hash=final_certificate.get("integrity_hash", ""),
        cert_id=fc_meta.get("cert_id", _cert_id("legacy", question_text)),
        timestamp=time.time(),
    )


# Convenience for server.py: the raw-baseline LLM call also remains importable.
def run_raw_baseline(
    resolved: ResolvedInput,
    backend: BackendChoice,
    api_key: str,
) -> tuple[str, dict]:
    """Side-by-side baseline: raw LLM call, no PCG-MAS. Returns (answer, meta)."""
    from .backends import call_chat
    evs = getattr(resolved, "evidence", []) or []
    evidence_block = "\n".join(
        f"- ({getattr(ev, 'publisher', getattr(ev, 'title', '?'))}) {getattr(ev, 'text', '')[:300]}"
        for ev in evs[:5]
    )
    question = getattr(resolved, "claim", "") or getattr(resolved, "question", "")
    prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{evidence_block}\n\n"
        f"Answer the question."
    )
    return call_chat(backend, api_key, prompt, max_tokens=512, temperature=0.0)


__all__ = [
    "run_pcg_mas",          # NEW master entrypoint, rich SSE events
    "run_pipeline",         # LEGACY backward-compat entrypoint
    "run_raw_baseline",
    "ChannelEvent",
    "FinalCertificate",
]
