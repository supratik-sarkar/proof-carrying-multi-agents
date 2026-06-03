"""Streaming PCG-MAS pipeline for the demo.

This is a SELF-CONTAINED, demo-grade implementation of the four-channel
verification pipeline (V_H proposer, V_Π executor, V_Γ judge, V_⊢ verifier).
It runs against any BYOK backend and any source-resolved input. It deliberately
mirrors the structure of src/pcg/orchestrator/langgraph_flow.py but avoids
importing the full pcg package (which pulls in torch and other heavy deps
that don't fit the HF Space CPU-basic image).

Each channel yields a structured event dict for the UI to render in real time.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Optional

from .backends import BackendChoice, call_chat
from .sources import ResolvedInput, Evidence


# ---------------------------------------------------------------------------
# Channel events the UI consumes
# ---------------------------------------------------------------------------

@dataclass
class ChannelEvent:
    channel: str             # 'V_H' | 'V_pi' | 'V_gamma' | 'V_entail' | 'verifier'
    state: str               # 'pending' | 'pass' | 'fail'
    verdict: str = ""        # short human-readable result
    detail: str = ""         # one-line explanation
    payload: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class FinalCertificate:
    accepted: bool
    claim: str
    answer: str
    channels: dict          # name -> ChannelEvent.as_dict()
    backend: dict
    source: dict
    integrity_hash: str
    cert_id: str
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------

PROPOSER_SYSTEM_WITH_EVIDENCE = (
    "You are V_H, the proposer in a Proof-Carrying Generation multi-agent system. "
    "Answer the user's QUESTION using ONLY the EVIDENCE provided. Cite the chunk "
    "indices you actually used.\n\n"
    "If the evidence appears stale, outdated, or insufficient for the question "
    "(e.g. the question asks about a CURRENT state but the evidence is dated, or "
    "the evidence is on a related-but-different topic), produce a SHORT honest "
    "answer that:\n"
    "  (a) states what the evidence DOES say (with the date or context),\n"
    "  (b) explicitly flags that the evidence may be outdated or insufficient "
    "for the user\'s actual question,\n"
    "  (c) sets confidence below 0.4.\n\n"
    "Only return a bare \"Insufficient evidence\" abstention if the evidence is "
    "TOTALLY unrelated to the question. Otherwise extract whatever the evidence "
    "DOES support and flag the staleness in the answer text itself.\n\n"
    "Respond in JSON exactly: "
    "{\"answer\": \"...\", \"cited_chunks\": [int,...], \"confidence\": float in [0,1]}"
)

PROPOSER_SYSTEM_NO_EVIDENCE = (
    "You are V_H, the proposer in a Proof-Carrying Generation multi-agent system. "
    "The user has NOT supplied external evidence. Answer the QUESTION from your "
    "own parametric knowledge. Be concrete and direct \u2014 a downstream judge "
    "will independently verify entailment, so abstaining costs the user a real "
    "answer. Only refuse if you are genuinely uncertain.\n\n"
    "Use \"cited_chunks\": [] since there is no external evidence. Respond in "
    "JSON exactly: "
    "{\"answer\": \"...\", \"cited_chunks\": [], \"confidence\": float in [0,1]}"
)

JUDGE_SYSTEM_NO_EVIDENCE = (
    "You are V_Γ, the entailment judge in a Proof-Carrying Generation system. "
    "The user did NOT supply external evidence; the proposer answered from its own "
    "parametric knowledge. Your job: act as a plausibility check.\n\n"
    "Reject (entailed=false) if any of these hold:\n"
    "  - The answer is a refusal or 'I don't know' / 'insufficient information' \u2014 "
    "    set is_abstention=true.\n"
    "  - The answer is internally inconsistent or self-contradictory.\n"
    "  - The answer is off-topic relative to the question.\n"
    "  - The answer makes a claim that is well-known to be false (e.g. wrong famous "
    "    historical fact, wrong capital city, wrong year by a wide margin).\n\n"
    "Accept (entailed=true) if the answer is a concrete, on-topic claim that is "
    "consistent with widely-known facts. You are NOT certifying the answer against "
    "any specific evidence \u2014 you are confirming it is a reasonable parametric "
    "response. The certificate will state 'parametric-only, no external evidence'.\n\n"
    "Respond in JSON exactly: "
    "{\"entailed\": true|false, \"reason\": \"...\", \"score\": float in [0,1], "
    "\"is_abstention\": true|false}"
)


JUDGE_SYSTEM = (
    "You are V_Γ, the entailment judge in a Proof-Carrying Generation system. "
    "Your job: decide whether the proposed ANSWER is (a) a real answer to the QUESTION "
    "and (b) supported by the EVIDENCE.\n\n"
    "Reject (entailed=false) if any of the following hold:\n"
    "  - The answer is a refusal, abstention, or 'insufficient evidence' / 'I don't know' "
    "    statement. Such answers do not entail anything; they decline to answer.\n"
    "  - The answer is a real claim but is not supported by the evidence.\n"
    "  - The answer contradicts the evidence.\n"
    "  - The answer is off-topic relative to the question.\n\n"
    "Accept (entailed=true) only if the answer makes a concrete claim that the evidence "
    "directly supports. The 'score' field is your confidence in [0,1].\n\n"
    "Respond in JSON exactly: "
    "{\"entailed\": true|false, \"reason\": \"...\", \"score\": float in [0,1], "
    "\"is_abstention\": true|false}"
)


def _proposer_prompt(claim: str, evs: list[Evidence]) -> str:
    evidence_block = "\n".join(
        f"[chunk {i}] ({ev.title}) {ev.text}" for i, ev in enumerate(evs)
    )
    return (
        f"QUESTION:\n{claim}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"Respond as JSON with keys: answer, cited_chunks, confidence."
    )


def _judge_prompt(claim: str, answer: str, evs: list[Evidence]) -> str:
    evidence_block = "\n".join(
        f"[chunk {i}] ({ev.title}) {ev.text}" for i, ev in enumerate(evs)
    )
    return (
        f"QUESTION:\n{claim}\n\n"
        f"PROPOSED ANSWER:\n{answer}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"Is the proposed answer entailed by the evidence? Respond as JSON "
        f"with keys: entailed, reason, score."
    )


def _safe_json(text: str) -> Optional[dict]:
    """Extract first JSON object from a (possibly fenced) LLM response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            return json.loads(cleaned)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Main pipeline — a generator yielding events
# ---------------------------------------------------------------------------

def run_pipeline(
    resolved: ResolvedInput,
    backend: BackendChoice,
    api_key: str,
    *,
    replay_check: bool = True,
    top_k_evidence: int = 5,
) -> Iterator[ChannelEvent | FinalCertificate]:
    """Run the four channels in sequence. Yields ChannelEvent for each phase
    transition (pending then pass/fail), and a final FinalCertificate.
    """

    # ---- Setup: cap evidence at top_k for context window safety ----
    from .sources import top_k as topk_fn
    resolved = topk_fn(resolved, top_k_evidence)

    # ============== Channel V_H: Proposer ==============
    # Choose proposer mode based on whether the user supplied real evidence.
    # A single placeholder evidence chunk (e.g. "(no evidence supplied)") counts
    # as no-evidence and we let the proposer use its parametric knowledge.
    has_evidence = bool(resolved.evidence) and not any(
        ev.publisher == "user_input" and ev.text.startswith("(no evidence")
        for ev in resolved.evidence
    )
    proposer_system = PROPOSER_SYSTEM_WITH_EVIDENCE if has_evidence else PROPOSER_SYSTEM_NO_EVIDENCE

    yield ChannelEvent(
        "V_H", "pending", "running",
        ("Asking proposer to answer from evidence…" if has_evidence
         else "No evidence supplied — proposer uses parametric knowledge."),
    )
    t0 = time.perf_counter()
    proposer_text, proposer_meta = call_chat(
        backend, api_key,
        _proposer_prompt(resolved.claim, resolved.evidence) if has_evidence
            else f"QUESTION:\n{resolved.claim}\n\nRespond as JSON with keys: answer, cited_chunks, confidence.",
        system=proposer_system, max_tokens=512, temperature=0.0,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0

    parsed = _safe_json(proposer_text)
    if not parsed or "answer" not in parsed:
        yield ChannelEvent(
            "V_H", "fail", "malformed",
            "Proposer returned no parseable JSON answer.",
            payload={"raw": proposer_text[:300], "meta": proposer_meta},
            elapsed_ms=elapsed,
        )
        # Cannot continue without an answer to verify
        yield FinalCertificate(
            accepted=False, claim=resolved.claim, answer="(no answer produced)",
            channels={"V_H": {"state": "fail", "reason": "proposer returned no JSON"}},
            backend=proposer_meta,
            source={"kind": resolved.source_kind, "label": resolved.source_label,
                    "n_evidence": len(resolved.evidence)},
            integrity_hash="", cert_id=_cert_id("fail", resolved.claim),
            timestamp=time.time(),
        )
        return

    answer = str(parsed.get("answer", "")).strip()
    cited = parsed.get("cited_chunks", []) or []
    conf = float(parsed.get("confidence", 0.0) or 0.0)

    # Validate cited chunks exist
    valid_cites = [c for c in cited if isinstance(c, int) and 0 <= c < len(resolved.evidence)]
    cite_ok = len(valid_cites) >= 1 if cited else (len(resolved.evidence) > 0)

    yield ChannelEvent(
        "V_H", "pass" if answer and cite_ok else "fail",
        f"answer ({len(answer)} chars), conf={conf:.2f}",
        f"Proposer produced an answer citing {len(valid_cites)} of {len(resolved.evidence)} chunks.",
        payload={"answer": answer, "cited_chunks": valid_cites,
                 "confidence": conf, "meta": proposer_meta},
        elapsed_ms=elapsed,
    )

    # ============== Channel V_Π: Executor (replay determinism) ==============
    yield ChannelEvent("V_pi", "pending", "running",
                       "Re-running the proposer to check replay determinism…")
    t0 = time.perf_counter()
    replay_ok = True
    replay_detail = "skipped (single-shot mode)"
    if replay_check:
        replay_text, _replay_meta = call_chat(
            backend, api_key,
            _proposer_prompt(resolved.claim, resolved.evidence) if has_evidence
                else f"QUESTION:\n{resolved.claim}\n\nRespond as JSON with keys: answer, cited_chunks, confidence.",
            system=proposer_system, max_tokens=512, temperature=0.0,
        )
        replay_parsed = _safe_json(replay_text)
        replay_answer = str(replay_parsed.get("answer", "")).strip() if replay_parsed else ""
        # We require lexically-similar (not necessarily identical) answers at T=0,
        # since most chat APIs don't guarantee bit-identity even at temperature=0.
        sim = _jaccard_similarity(answer, replay_answer)
        replay_ok = sim >= 0.5
        replay_detail = (f"replay similarity (Jaccard) = {sim:.2f}; "
                         f"accepted as deterministic" if replay_ok
                         else f"replay diverged (Jaccard = {sim:.2f}); flagging drift")
    elapsed = (time.perf_counter() - t0) * 1000.0
    yield ChannelEvent(
        "V_pi", "pass" if replay_ok else "fail",
        "deterministic" if replay_ok else "drift",
        replay_detail,
        payload={"replay_check": replay_check},
        elapsed_ms=elapsed,
    )

    # ============== Channel V_Γ: Judge (entailment OR plausibility) ==============
    judge_role = ("entailment judge verifying the answer against evidence"
                  if has_evidence else
                  "plausibility judge (no external evidence; parametric-only)")
    yield ChannelEvent("V_gamma", "pending", "running",
                       f"Asking {judge_role}…")
    t0 = time.perf_counter()
    if has_evidence:
        judge_prompt_text = _judge_prompt(resolved.claim, answer, resolved.evidence)
        judge_system = JUDGE_SYSTEM
    else:
        judge_prompt_text = (
            f"QUESTION:\n{resolved.claim}\n\n"
            f"PROPOSED ANSWER:\n{answer}\n\n"
            "Is this answer plausible and on-topic? Respond as JSON with keys: "
            "entailed, reason, score, is_abstention."
        )
        judge_system = JUDGE_SYSTEM_NO_EVIDENCE
    judge_text, judge_meta = call_chat(
        backend, api_key, judge_prompt_text,
        system=judge_system, max_tokens=256, temperature=0.0,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    j = _safe_json(judge_text) or {}
    entailed = bool(j.get("entailed", False))
    score = float(j.get("score", 0.0) or 0.0)
    reason = str(j.get("reason", "(no reason)"))[:240]
    judge_label = "entailed" if has_evidence else "plausible"
    yield ChannelEvent(
        "V_gamma", "pass" if entailed else "fail",
        f"{judge_label}={entailed}, score={score:.2f}",
        reason + (" (parametric-only — no external evidence)" if not has_evidence else ""),
        payload={"raw": judge_text[:400], "meta": judge_meta,
                 "mode": "entailment" if has_evidence else "plausibility"},
        elapsed_ms=elapsed,
    )

    # ============== Channel V_⊢: Verifier (integration) ==============
    yield ChannelEvent("V_entail", "pending", "running",
                       "Integrating channel verdicts into final accept/reject…")
    t0 = time.perf_counter()
    is_abstention = bool(j.get("is_abstention", False))
    # In no-evidence mode, citing is vacuously OK (proposer cites nothing by design).
    cite_required_pass = cite_ok if has_evidence else True
    accept = bool(
        answer
        and cite_required_pass
        and replay_ok
        and entailed
        and score >= 0.5
        and not is_abstention
    )
    rationale = _build_rationale(
        cite_ok=cite_required_pass, replay_ok=replay_ok, entailed=entailed,
        score=score, conf=conf, n_evidence=len(resolved.evidence),
        is_abstention=is_abstention,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    yield ChannelEvent(
        "V_entail", "pass" if accept else "fail",
        "accept" if accept else "reject",
        rationale,
        payload={
            "cite_ok": cite_ok, "replay_ok": replay_ok,
            "entailed": entailed, "judge_score": score,
            "proposer_conf": conf,
        },
        elapsed_ms=elapsed,
    )

    # ============== Final certificate ==============
    cert_payload = {
        "claim": resolved.claim,
        "answer": answer,
        "cited_chunks": valid_cites,
        "proposer_conf": conf,
        "replay_ok": replay_ok,
        "judge_mode": "entailment" if has_evidence else "plausibility",
        "judge_entailed": entailed,
        "judge_score": score,
        "n_evidence": len(resolved.evidence) if has_evidence else 0,
        "evidence_supplied": has_evidence,
        "backend": {"provider": backend.provider, "model": backend.model_id},
    }
    integrity = hashlib.sha256(
        json.dumps(cert_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    yield FinalCertificate(
        accepted=accept,
        claim=resolved.claim,
        answer=answer,
        channels={
            "V_H":      {"pass": bool(answer and cite_ok), "conf": conf,
                          "cited_chunks": valid_cites},
            "V_pi":     {"pass": replay_ok, "detail": replay_detail},
            "V_gamma":  {"pass": entailed, "score": score, "reason": reason},
            "V_entail": {"pass": accept, "rationale": rationale},
        },
        backend={"provider": backend.provider, "model": backend.model_id},
        source={"kind": resolved.source_kind, "label": resolved.source_label,
                "n_evidence": len(resolved.evidence)},
        integrity_hash=integrity,
        cert_id=_cert_id("accept" if accept else "reject", resolved.claim),
        timestamp=time.time(),
    )


def run_raw_baseline(
    resolved: ResolvedInput,
    backend: BackendChoice,
    api_key: str,
) -> tuple[str, dict]:
    """Side-by-side baseline: raw LLM call, no PCG-MAS. Returns (answer, meta)."""
    evidence_block = "\n".join(
        f"- ({ev.title}) {ev.text[:300]}" for ev in resolved.evidence[:5]
    )
    prompt = (
        f"Question: {resolved.claim}\n\n"
        f"Context:\n{evidence_block}\n\n"
        f"Answer the question."
    )
    return call_chat(backend, api_key, prompt, max_tokens=512, temperature=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _build_rationale(*, cite_ok, replay_ok, entailed, score, conf, n_evidence,
                     is_abstention: bool = False) -> str:
    parts = []
    if is_abstention:
        parts.append("proposer abstained (no concrete answer) — the judge flagged "
                     "this as a non-answer rather than a verifiable claim")
    if not cite_ok:
        parts.append("proposer failed to cite valid evidence chunks")
    if not replay_ok:
        parts.append("replay diverged — answer is not deterministic")
    if not entailed:
        parts.append(f"judge rejected entailment (score={score:.2f})")
    if entailed and score < 0.5:
        parts.append(f"judge entailment score below threshold ({score:.2f} < 0.50)")
    if not parts:
        return (f"All channels pass: cited evidence, replay deterministic, "
                f"judge entailment {score:.2f}.")
    return "Rejected because: " + "; ".join(parts) + "."


def _cert_id(decision: str, claim: str) -> str:
    seed = f"{decision}:{claim}:{time.time()}".encode("utf-8")
    return f"cert-{hashlib.sha256(seed).hexdigest()[:12]}"
