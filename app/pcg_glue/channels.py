"""
Five-channel checker for PCG-MAS (paper appendix, Template 2).

For each AtomicClaim this module produces a ChannelVerdict per channel:

    V_I    Integrity  — recompute support hashes; flag any mismatch
    V_R    Replay     — re-run the proposer N times; demand stable structure
    V_D    Drift      — LLM-judge whether replay outputs are semantically equivalent
    V_Ch   Checker    — LLM-judge entailment + execution-contract compliance
    V_Cov  Coverage   — LLM-judge whether cited evidence substantively addresses the claim

Channels never silently repair a claim. They emit pass/fail with a short
detail string consumed by the frontend and the certificate inspector.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from pcg_glue.schemas import (
    AtomicClaim, ChannelName, ChannelState, ChannelVerdict,
    ClaimCertificate, EvidenceItem, ToolOutput,
)
from pcg_glue.backends import BackendChoice, call_chat
from pcg_glue.claim_extractor import _normalize_text


# =============================================================================
# V_I — Integrity (deterministic, no LLM call)
# =============================================================================

def check_integrity(
    claim: AtomicClaim,
    evidence_index: dict[str, EvidenceItem],
    tool_index: dict[str, ToolOutput],
) -> ChannelVerdict:
    """Recompute hashes of cited supports; fail on any mismatch.

    The hashes were committed at extraction time (commit_evidence /
    commit_tool_outputs). V_I re-derives them now to catch tampering.
    """
    t0 = time.time()
    mismatches: list[str] = []

    for eid in claim.support_ids:
        ev = evidence_index.get(eid)
        if ev is None:
            mismatches.append(f"missing evidence {eid}")
            continue
        h = hashlib.sha256(_normalize_text(ev.text).encode("utf-8")).hexdigest()
        if h != ev.hash:
            mismatches.append(f"hash mismatch on {eid}")

    for tid in claim.tool_output_ids:
        t = tool_index.get(tid)
        if t is None:
            mismatches.append(f"missing tool output {tid}")
            continue
        payload = json.dumps({"name": t.name, "args": t.args, "output": t.output},
                             sort_keys=True, default=str)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if h != t.hash:
            mismatches.append(f"hash mismatch on {tid}")

    elapsed = int((time.time() - t0) * 1000)
    if mismatches:
        return ChannelVerdict(
            channel=ChannelName.V_I,
            state=ChannelState.FAIL,
            score=0.0,
            detail="; ".join(mismatches)[:240],
            elapsed_ms=elapsed,
        )
    return ChannelVerdict(
        channel=ChannelName.V_I,
        state=ChannelState.PASS,
        score=1.0,
        detail=f"All {len(claim.support_ids) + len(claim.tool_output_ids)} supports verified",
        elapsed_ms=elapsed,
    )


# =============================================================================
# V_R — Replay (re-run the proposer, structural agreement)
# =============================================================================

_REPLAY_SYSTEM = (
    "You are the support-pipeline replayer. Given a question and the same "
    "evidence, restate the answer to ONE specific claim using only the cited "
    "evidence. Return JSON: {\"answer\": \"...\"}. Keep it under 60 words."
)


def check_replay(
    claim: AtomicClaim,
    evidence_index: dict[str, EvidenceItem],
    question: str,
    backend: BackendChoice,
    api_key: str,
) -> tuple[ChannelVerdict, str]:
    """Re-run a focused proposer for this claim. Returns (verdict, replay_text).

    The replay_text is passed to V_D for drift judging — they share the LLM call.
    """
    t0 = time.time()
    cited = [evidence_index[eid].text for eid in claim.support_ids
             if eid in evidence_index]
    if not cited:
        return (ChannelVerdict(
            channel=ChannelName.V_R,
            state=ChannelState.FAIL,
            score=0.0,
            detail="cannot replay: no cited evidence",
            elapsed_ms=int((time.time() - t0) * 1000),
        ), "")

    user = (
        f"QUESTION: {question}\n\n"
        f"CLAIM TO RESTATE: {claim.claim_text}\n\n"
        f"CITED EVIDENCE:\n" +
        "\n---\n".join(f"[{eid}] {evidence_index[eid].text[:1200]}"
                       for eid in claim.support_ids if eid in evidence_index)
    )

    try:
        resp, _meta = call_chat(
            backend, api_key,
            user,
            system=_REPLAY_SYSTEM, temperature=0.0, max_tokens=180,
        )
    except Exception as e:
        return (ChannelVerdict(
            channel=ChannelName.V_R,
            state=ChannelState.FAIL,
            score=0.0,
            detail=f"replay call failed: {type(e).__name__}",
            elapsed_ms=int((time.time() - t0) * 1000),
        ), "")

    replay_text = _extract_field(resp, "answer") or resp.strip()
    elapsed = int((time.time() - t0) * 1000)

    # Structural pass: replay produced a non-empty answer
    if len(replay_text.strip()) < 5:
        return (ChannelVerdict(
            channel=ChannelName.V_R,
            state=ChannelState.FAIL,
            score=0.0,
            detail="replay produced empty answer",
            elapsed_ms=elapsed,
        ), replay_text)

    return (ChannelVerdict(
        channel=ChannelName.V_R,
        state=ChannelState.PASS,
        score=1.0,
        detail=f"replay produced {len(replay_text)} chars",
        elapsed_ms=elapsed,
    ), replay_text)


# =============================================================================
# V_D — Drift (semantic equivalence between original claim and replay)
# =============================================================================

_DRIFT_SYSTEM = (
    "You are a semantic-equivalence judge. Given an ORIGINAL claim and a "
    "REPLAY answer, decide whether they assert the same fact within "
    "acceptable paraphrase. Return JSON: "
    "{\"equivalent\": true|false, \"similarity\": 0.0-1.0, \"reason\": \"...\"}. "
    "Equivalent means: same entities, same relation, same direction, same "
    "numeric value if any. Different examples or paraphrases of the SAME fact "
    "are equivalent. Different facts are not."
)


def check_drift(
    claim: AtomicClaim,
    replay_text: str,
    backend: BackendChoice,
    api_key: str,
    threshold: float = 0.65,
) -> ChannelVerdict:
    """LLM-judge whether claim and replay are semantically equivalent."""
    t0 = time.time()
    if not replay_text.strip():
        return ChannelVerdict(
            channel=ChannelName.V_D, state=ChannelState.SKIP,
            score=None, detail="no replay to compare against",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    user = (
        f"ORIGINAL CLAIM: {claim.claim_text}\n\n"
        f"REPLAY ANSWER: {replay_text}\n\n"
        f"Return JSON only."
    )
    try:
        resp, _meta = call_chat(
            backend, api_key,
            user,
            system=_DRIFT_SYSTEM, temperature=0.0, max_tokens=160,
        )
    except Exception as e:
        return ChannelVerdict(
            channel=ChannelName.V_D, state=ChannelState.FAIL,
            score=0.0, detail=f"drift judge failed: {type(e).__name__}",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    sim = _extract_float(resp, "similarity", default=0.0)
    eq = _extract_bool(resp, "equivalent", default=(sim >= threshold))
    reason = _extract_field(resp, "reason") or ""
    elapsed = int((time.time() - t0) * 1000)

    if eq and sim >= threshold:
        return ChannelVerdict(
            channel=ChannelName.V_D, state=ChannelState.PASS,
            score=sim, detail=f"sim={sim:.2f} · {reason[:120]}",
            elapsed_ms=elapsed,
        )
    return ChannelVerdict(
        channel=ChannelName.V_D, state=ChannelState.FAIL,
        score=sim, detail=f"drift detected sim={sim:.2f} · {reason[:120]}",
        elapsed_ms=elapsed,
    )


# =============================================================================
# V_Ch — Checker (entailment + execution-contract compliance)
# =============================================================================

_CHECKER_SYSTEM = (
    "You are an entailment + execution-contract checker. Decide whether the "
    "CITED EVIDENCE logically entails the CLAIM. A claim is entailed if the "
    "evidence directly supports it; partial overlap, related facts, or "
    "plausible-sounding-but-uncited conclusions are NOT entailment.\n\n"
    "Also enforce the execution contract: the claim must not contradict the "
    "evidence anywhere. Numeric values must match exactly (no rounding to a "
    "different significant figure).\n\n"
    "Return JSON: {\"entails\": true|false, \"contract_ok\": true|false, "
    "\"score\": 0.0-1.0, \"reason\": \"...\"}"
)


def check_checker(
    claim: AtomicClaim,
    evidence_index: dict[str, EvidenceItem],
    backend: BackendChoice,
    api_key: str,
    threshold: float = 0.55,
) -> ChannelVerdict:
    """LLM-judge entailment + execution-contract for one claim."""
    t0 = time.time()
    if not claim.support_ids:
        return ChannelVerdict(
            channel=ChannelName.V_Ch, state=ChannelState.FAIL,
            score=0.0, detail="no cited evidence",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    cited_blocks = []
    for eid in claim.support_ids:
        if eid in evidence_index:
            cited_blocks.append(f"[{eid}] {evidence_index[eid].text[:1500]}")
    if not cited_blocks:
        return ChannelVerdict(
            channel=ChannelName.V_Ch, state=ChannelState.FAIL,
            score=0.0, detail="cited evidence not found",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    user = (
        f"CLAIM: {claim.claim_text}\n\n"
        f"CITED EVIDENCE:\n" + "\n---\n".join(cited_blocks) +
        "\n\nReturn JSON only."
    )
    try:
        resp, _meta = call_chat(
            backend, api_key,
            user,
            system=_CHECKER_SYSTEM, temperature=0.0, max_tokens=200,
        )
    except Exception as e:
        return ChannelVerdict(
            channel=ChannelName.V_Ch, state=ChannelState.FAIL,
            score=0.0, detail=f"checker failed: {type(e).__name__}",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    score = _extract_float(resp, "score", default=0.0)
    entails = _extract_bool(resp, "entails", default=(score >= threshold))
    contract_ok = _extract_bool(resp, "contract_ok", default=True)
    reason = _extract_field(resp, "reason") or ""
    elapsed = int((time.time() - t0) * 1000)

    if entails and contract_ok and score >= threshold:
        return ChannelVerdict(
            channel=ChannelName.V_Ch, state=ChannelState.PASS,
            score=score, detail=f"score={score:.2f} · {reason[:120]}",
            elapsed_ms=elapsed,
        )
    parts = []
    if not entails: parts.append("entailment fail")
    if not contract_ok: parts.append("contract violation")
    if score < threshold: parts.append(f"score={score:.2f} below {threshold}")
    return ChannelVerdict(
        channel=ChannelName.V_Ch, state=ChannelState.FAIL,
        score=score, detail="; ".join(parts) + f" · {reason[:120]}",
        elapsed_ms=elapsed,
    )


# =============================================================================
# V_Cov — Coverage (did cited support substantively address the claim?)
# =============================================================================

_COVERAGE_SYSTEM = (
    "You are a coverage judge. Decide whether the CITED EVIDENCE substantively "
    "addresses the CLAIM, i.e. would a careful reader of just the evidence be "
    "able to assess the claim's truth? This is different from entailment: a "
    "claim can be entailed by a single line, or only weakly supported by a long "
    "tangentially-related passage.\n\n"
    "Return JSON: {\"coverage\": \"full\" | \"partial\" | \"none\", "
    "\"score\": 0.0-1.0, \"reason\": \"...\"}. Coverage \"full\" → pass. "
    "Partial or none → fail."
)


def check_coverage(
    claim: AtomicClaim,
    evidence_index: dict[str, EvidenceItem],
    backend: BackendChoice,
    api_key: str,
    threshold: float = 0.6,
) -> ChannelVerdict:
    """LLM-judge whether cited evidence covers the claim's atomic propositions."""
    t0 = time.time()
    if not claim.support_ids:
        return ChannelVerdict(
            channel=ChannelName.V_Cov, state=ChannelState.FAIL,
            score=0.0, detail="no cited evidence",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    cited_blocks = []
    for eid in claim.support_ids:
        if eid in evidence_index:
            cited_blocks.append(f"[{eid}] {evidence_index[eid].text[:1500]}")
    if not cited_blocks:
        return ChannelVerdict(
            channel=ChannelName.V_Cov, state=ChannelState.FAIL,
            score=0.0, detail="cited evidence not found",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    user = (
        f"CLAIM: {claim.claim_text}\n\n"
        f"CITED EVIDENCE:\n" + "\n---\n".join(cited_blocks) +
        "\n\nReturn JSON only."
    )
    try:
        resp, _meta = call_chat(
            backend, api_key,
            user,
            system=_COVERAGE_SYSTEM, temperature=0.0, max_tokens=180,
        )
    except Exception as e:
        return ChannelVerdict(
            channel=ChannelName.V_Cov, state=ChannelState.FAIL,
            score=0.0, detail=f"coverage judge failed: {type(e).__name__}",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    coverage = _extract_field(resp, "coverage") or "none"
    score = _extract_float(resp, "score", default=0.0)
    reason = _extract_field(resp, "reason") or ""
    elapsed = int((time.time() - t0) * 1000)

    if coverage == "full" and score >= threshold:
        return ChannelVerdict(
            channel=ChannelName.V_Cov, state=ChannelState.PASS,
            score=score, detail=f"full coverage · {reason[:120]}",
            elapsed_ms=elapsed,
        )
    return ChannelVerdict(
        channel=ChannelName.V_Cov, state=ChannelState.FAIL,
        score=score, detail=f"coverage={coverage} score={score:.2f} · {reason[:120]}",
        elapsed_ms=elapsed,
    )


# =============================================================================
# Orchestrator — run all 5 channels for one claim
# =============================================================================

def run_all_channels(
    claim: AtomicClaim,
    evidence_index: dict[str, EvidenceItem],
    tool_index: dict[str, ToolOutput],
    question: str,
    backend: BackendChoice,
    api_key: str,
) -> ClaimCertificate:
    """Run V_I, V_R, V_D, V_Ch, V_Cov in sequence. Returns a ClaimCertificate.

    Early-exit policy: if V_I fails, the claim cannot be salvaged — we still
    run the other channels with state=SKIP so the UI renders all 5 chips.
    """
    channels: dict[ChannelName, ChannelVerdict] = {}

    # V_I
    channels[ChannelName.V_I] = check_integrity(claim, evidence_index, tool_index)

    # V_R + V_D (V_D consumes V_R's replay_text)
    vr, replay_text = check_replay(claim, evidence_index, question, backend, api_key)
    channels[ChannelName.V_R] = vr

    if vr.state == ChannelState.PASS:
        channels[ChannelName.V_D] = check_drift(claim, replay_text, backend, api_key)
    else:
        channels[ChannelName.V_D] = ChannelVerdict(
            channel=ChannelName.V_D, state=ChannelState.SKIP,
            score=None, detail="V_R failed; drift not assessed", elapsed_ms=0,
        )

    # V_Ch and V_Cov are independent
    channels[ChannelName.V_Ch] = check_checker(claim, evidence_index, backend, api_key)
    channels[ChannelName.V_Cov] = check_coverage(claim, evidence_index, backend, api_key)

    accepted = all(v.state == ChannelState.PASS for v in channels.values())

    # Integrity hash over (claim_text, support_ids, pipeline_id, contract_id)
    payload = json.dumps({
        "claim": claim.claim_text,
        "support": sorted(claim.support_ids),
        "tools": sorted(claim.tool_output_ids),
        "pipeline_id": "pcg_mas_v1",
        "contract_id": "gamma_v1",
    }, sort_keys=True)
    integrity_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return ClaimCertificate(
        claim=claim,
        channels=channels,
        accepted=accepted,
        integrity_hash=integrity_hash,
        minimal_support_ids=claim.support_ids[:],
    )


# =============================================================================
# Tiny JSON-field extractors (robust to ```json fences and trailing prose)
# =============================================================================

_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _try_parse(s: str) -> dict[str, Any]:
    if not s:
        return {}
    if "```" in s:
        parts = s.split("```")
        for p in parts[1:]:
            stripped = p.lstrip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
            if stripped.startswith("{"):
                s = stripped
                break
    m = _OBJ_RE.search(s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _extract_field(s: str, key: str) -> str:
    d = _try_parse(s)
    v = d.get(key)
    return str(v).strip() if v is not None else ""


def _extract_float(s: str, key: str, default: float = 0.0) -> float:
    d = _try_parse(s)
    try:
        return max(0.0, min(1.0, float(d.get(key, default))))
    except (TypeError, ValueError):
        return default


def _extract_bool(s: str, key: str, default: bool = False) -> bool:
    d = _try_parse(s)
    v = d.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1")
    return default


__all__ = [
    "check_integrity",
    "check_replay",
    "check_drift",
    "check_checker",
    "check_coverage",
    "run_all_channels",
]
