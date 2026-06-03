"""
Atomic claim extraction (paper appendix, Template 1).

Given (question, retrieved_evidence, tool_outputs, backend, api_key) this
module produces:
    - answer_draft : str             — the model's initial answer
    - claims       : list[AtomicClaim]   — split into atomic, evidence-cited units

Every claim cites at least one evidence_id or tool_output_id. Claims that
cite nothing are kept (with uncertainty_flags=["missing_evidence"]) so V_Cov
can fail them downstream — silent dropping would be dishonest.

The proposer is forced into structured JSON output. Parsing is defensive:
malformed JSON triggers a single retry with a stricter "JSON only" prefix.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from pcg_glue.schemas import (
    AtomicClaim, EvidenceItem, ToolOutput,
)
from pcg_glue.backends import BackendChoice, call_chat


# -----------------------------------------------------------------------------
# Prompts (released under the paper artifacts URL; kept here for in-process use)
# -----------------------------------------------------------------------------

CLAIM_EXTRACTOR_SYSTEM = (
    "You are a claim generator for a proof-carrying agent.\n"
    "Do not emit unsupported claims. Split your answer into atomic claims.\n"
    "Every claim must cite at least one evidence_id (e1, e2, ...) or "
    "tool_output_id (t1, t2, ...). If you have no evidence for a sub-claim, "
    "omit it from the answer rather than fabricating.\n"
    "\n"
    "Respond with a SINGLE JSON object, no prose before or after. The exact "
    "shape is:\n"
    "{\n"
    '  "answer_draft": "<one-paragraph answer to the question>",\n'
    '  "claims": [\n'
    "    {\n"
    '      "claim_id": "c1",\n'
    '      "claim_text": "<one atomic factual claim>",\n'
    '      "support_ids": ["e1", "e2"],\n'
    '      "tool_output_ids": [],\n'
    '      "confidence": 0.0,\n'
    '      "uncertainty_flags": ["none"]\n'
    "    }\n"
    "  ],\n"
    '  "abstain_reason": null\n'
    "}\n"
    "\n"
    "Allowed uncertainty_flags: missing_evidence, numeric_ambiguity, "
    "tool_dependency, none. confidence is in [0, 1]. claim_ids must be c1, "
    "c2, c3, ... in order.\n"
)


def _build_user_prompt(
    question: str,
    evidence: list[EvidenceItem],
    tool_outputs: list[ToolOutput],
) -> str:
    parts = [f"QUESTION:\n{question.strip()}\n"]
    if evidence:
        parts.append("\nRETRIEVED EVIDENCE:")
        for ev in evidence:
            parts.append(
                f"\n[{ev.id}] (source: {ev.source})\n{ev.text.strip()[:1600]}"
            )
    else:
        parts.append("\nRETRIEVED EVIDENCE: (none — answer from parametric knowledge "
                     "ONLY if you are highly confident; otherwise abstain)")
    if tool_outputs:
        parts.append("\n\nTOOL OUTPUTS:")
        for t in tool_outputs:
            parts.append(f"\n[{t.id}] {t.name}({json.dumps(t.args)})\n{t.output[:800]}")
    return "\n".join(parts)


# -----------------------------------------------------------------------------
# JSON extraction
# -----------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response. Tolerant of
    ```json fences and leading prose."""
    if not text:
        raise ValueError("empty model response")
    # Strip ```json fences if present
    if "```" in text:
        # Take content between first pair of fences
        parts = text.split("```")
        for i, p in enumerate(parts):
            if i == 0:
                continue
            stripped = p.lstrip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
            if stripped.startswith("{"):
                text = stripped
                break
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object found in response (head: {text[:200]!r})")
    return json.loads(m.group(0))


# -----------------------------------------------------------------------------
# Claim list validation
# -----------------------------------------------------------------------------

_ALLOWED_FLAGS = {"missing_evidence", "numeric_ambiguity", "tool_dependency", "none"}


def _normalize_claim(
    raw: dict[str, Any],
    idx: int,
    valid_evidence_ids: set[str],
    valid_tool_ids: set[str],
) -> AtomicClaim:
    cid = raw.get("claim_id") or f"c{idx + 1}"
    text = (raw.get("claim_text") or "").strip()
    if not text:
        # Skip empty claims later
        text = "(empty)"
    sids = [s for s in (raw.get("support_ids") or []) if s in valid_evidence_ids]
    tids = [t for t in (raw.get("tool_output_ids") or []) if t in valid_tool_ids]
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    flags = [f for f in (raw.get("uncertainty_flags") or []) if f in _ALLOWED_FLAGS]
    # If support is missing, add the flag explicitly. Drop any "none" since
    # "none" must be mutually exclusive with every real flag.
    if not sids and not tids:
        flags = [f for f in flags if f != "none"]
        if "missing_evidence" not in flags:
            flags.append("missing_evidence")
    # If any real flag is present, drop "none"
    real_flags = [f for f in flags if f != "none"]
    if real_flags:
        flags = real_flags
    elif not flags:
        flags = ["none"]
    return AtomicClaim(
        claim_id=cid,
        claim_text=text,
        support_ids=sids,
        tool_output_ids=tids,
        confidence=conf,
        uncertainty_flags=flags,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def extract_claims(
    question: str,
    evidence: list[EvidenceItem],
    tool_outputs: list[ToolOutput],
    backend: BackendChoice,
    api_key: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> tuple[str, list[AtomicClaim], dict[str, Any]]:
    """Run claim extraction. Returns (answer_draft, claims, raw_meta).

    raw_meta carries token counts and elapsed time for the audit trail.
    """
    user_prompt = _build_user_prompt(question, evidence, tool_outputs)
    valid_evidence = {ev.id for ev in evidence}
    valid_tools = {t.id for t in tool_outputs}

    t0 = time.time()
    response, meta = call_chat(
            backend, api_key,
            user_prompt,
            system=CLAIM_EXTRACTOR_SYSTEM, temperature=temperature, max_tokens=max_tokens,
        )

    try:
        parsed = _extract_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        # One stricter retry
        retry_response, retry_meta = call_chat(
            backend, api_key,
            user_prompt + "\n\nReturn ONLY a JSON object. No prose, no markdown fences.",
            system=CLAIM_EXTRACTOR_SYSTEM,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        meta["tokens_in"]  = meta.get("tokens_in", 0)  + retry_meta.get("tokens_in", 0)
        meta["tokens_out"] = meta.get("tokens_out", 0) + retry_meta.get("tokens_out", 0)
        try:
            parsed = _extract_json(retry_response)
        except (ValueError, json.JSONDecodeError) as e2:
            # Final fallback: treat the whole response as a single uncited claim
            return (
                response,
                [AtomicClaim(
                    claim_id="c1",
                    claim_text=response.strip()[:500] or "(empty)",
                    support_ids=[],
                    tool_output_ids=[],
                    confidence=0.0,
                    uncertainty_flags=["missing_evidence"],
                )],
                {**meta, "parse_error": str(e2), "elapsed_ms": int((time.time() - t0) * 1000)},
            )

    answer_draft = (parsed.get("answer_draft") or "").strip()
    if parsed.get("abstain_reason"):
        return (
            f"(abstained: {parsed['abstain_reason']})",
            [],
            {**meta, "abstain_reason": parsed["abstain_reason"],
             "elapsed_ms": int((time.time() - t0) * 1000)},
        )

    raw_claims = parsed.get("claims") or []
    if not isinstance(raw_claims, list):
        raw_claims = []
    claims: list[AtomicClaim] = []
    for i, rc in enumerate(raw_claims):
        if not isinstance(rc, dict):
            continue
        c = _normalize_claim(rc, i, valid_evidence, valid_tools)
        if c.claim_text != "(empty)":
            claims.append(c)

    # Re-number c1, c2, ... in order
    for i, c in enumerate(claims):
        c.claim_id = f"c{i + 1}"

    meta["elapsed_ms"] = int((time.time() - t0) * 1000)
    return answer_draft, claims, meta


# -----------------------------------------------------------------------------
# Evidence / tool-output commitment helpers (used by V_I)
# -----------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    """Whitespace-canonical text for hashing."""
    return " ".join(s.split())


def commit_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Return new EvidenceItem list with hash filled in. Pure / idempotent."""
    out = []
    for ev in items:
        h = hashlib.sha256(_normalize_text(ev.text).encode("utf-8")).hexdigest()
        out.append(EvidenceItem(
            id=ev.id, text=ev.text, source=ev.source, hash=h, span=ev.span,
        ))
    return out


def commit_tool_outputs(items: list[ToolOutput]) -> list[ToolOutput]:
    out = []
    for t in items:
        payload = json.dumps({"name": t.name, "args": t.args, "output": t.output},
                             sort_keys=True, default=str)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        out.append(ToolOutput(
            id=t.id, name=t.name, args=t.args, output=t.output, hash=h,
        ))
    return out


__all__ = [
    "extract_claims",
    "commit_evidence",
    "commit_tool_outputs",
    "CLAIM_EXTRACTOR_SYSTEM",
]
