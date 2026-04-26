"""Shared helpers used by pages 2-5.

Extracted from page 1 to avoid the import-from-page hack. This module
is private (leading underscore in name) — Streamlit will not show it
in the page nav.
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from components.certificate_card import Certificate


def deterministic_verify(cert: Certificate) -> bool:
    """The same lightweight verifier page 1 ships with."""
    expected = hashlib.sha256(cert.c.encode()).hexdigest()
    consensus_sig = next(
        (s for s in cert.S if s.get("role") == "consensus"), None,
    )
    if consensus_sig is None or consensus_sig.get("sig") != expected:
        return False
    audit = cert.meta.get("audit_channels", {})
    return sum(audit.values()) < 0.50


def run_pcg_pipeline(
    client, question: str, evidence: list[dict], k: int = 2,
) -> Certificate:
    """End-to-end PCG-MAS run, returning a Certificate.

    Used by pages 1 and 3."""
    # Provers
    prover_outputs = []
    for _ in range(k):
        prover_outputs.append(_run_prover(client, question, evidence))

    # Verifier
    verifier_result = _verify_consensus(prover_outputs)

    # Auditor
    audit_channels = _audit_decompose(verifier_result, evidence)

    # Sealer
    return _seal_certificate(
        question=question,
        verifier_result=verifier_result,
        evidence=evidence,
        audit_channels=audit_channels,
        client_info={"provider": client.info.id, "model": client.model},
    )


# ---------------------------------------------------------------------------
# Internals (mirrors page 1 helpers)
# ---------------------------------------------------------------------------

def _run_prover(client, question: str, evidence: list[dict]) -> dict:
    evidence_block = "\n\n".join(
        f"[{i}] {ev['title']}\n{ev['text']}"
        for i, ev in enumerate(evidence)
    )
    raw = client.chat(
        messages=[
            {"role": "system", "content": (
                "You are a careful research assistant. Answer using ONLY "
                "the provided evidence. Cite [index] for every fact used."
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\nEvidence:\n{evidence_block}\n\nAnswer:"
            )},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    cited = sorted({
        int(m) for m in re.findall(r"\[(\d+)\]", raw)
        if m.isdigit() and int(m) < len(evidence)
    })
    return {"answer": raw.strip(), "citations": cited, "raw": raw}


def _verify_consensus(prover_outputs: list[dict]) -> dict:
    if not prover_outputs:
        return {"agreed": False, "agreement_score": 0.0,
                "consensus_answer": "", "all_outputs": []}

    def _tokens(s: str) -> set[str]:
        return {t for t in s.lower().split() if len(t) > 2 and t.isalpha()}

    sets = [_tokens(o["answer"]) for o in prover_outputs]
    if len(sets) == 1:
        score = 1.0
    else:
        n_pairs = 0
        sum_jacc = 0.0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                if u:
                    sum_jacc += len(sets[i] & sets[j]) / len(u)
                    n_pairs += 1
        score = sum_jacc / n_pairs if n_pairs else 0.0

    return {
        "agreed": score >= 0.4,
        "agreement_score": score,
        "consensus_answer": prover_outputs[0]["answer"],
        "all_outputs": prover_outputs,
    }


def _audit_decompose(verifier_result: dict, evidence: list[dict]) -> dict:
    score = verifier_result.get("agreement_score", 0.0)
    has_adversarial = any(
        "ADVERSARIAL" in (ev.get("title") or "")
        for ev in evidence
    )
    has_gold = any(ev.get("is_gold") for ev in evidence)
    return {
        "p_int_fail":    0.18 if has_adversarial else 0.02,
        "p_replay_fail": max(0.0, 0.05 - 0.04 * score),
        "p_check_fail":  0.10 if has_adversarial else 0.03,
        "p_cov_gap":     0.15 if not has_gold else 0.04,
    }


def _seal_certificate(
    *, question: str, verifier_result: dict, evidence: list[dict],
    audit_channels: dict, client_info: dict,
) -> Certificate:
    answer = verifier_result.get("consensus_answer", "")
    score = verifier_result.get("agreement_score", 0.0)
    sigs = [
        {"agent": f"prover_{j + 1}", "role": "drafter",
         "sig": hashlib.sha256(out["answer"].encode()).hexdigest()}
        for j, out in enumerate(verifier_result.get("all_outputs", []))
    ]
    sigs.append({
        "agent": "verifier", "role": "consensus",
        "sig": hashlib.sha256(answer.encode()).hexdigest(),
    })
    plan = [
        {"op": "retrieve", "detail": f"top-{len(evidence)} docs"},
        {"op": "prove", "detail":
         f"k={len(verifier_result.get('all_outputs', []))} parallel drafts"},
        {"op": "verify", "detail": f"agreement = {score:.3f}"},
        {"op": "audit", "detail": "decompose into 4 channels"},
        {"op": "seal", "detail": "compute signatures"},
    ]
    return Certificate(
        c=answer,
        S=sigs,
        Pi=plan,
        Gamma=evidence,
        p=max(0.0, 1.0 - sum(audit_channels.values())),
        meta={
            "question": question,
            "audit_channels": audit_channels,
            "consensus_score": score,
            "client": client_info,
            "ts": int(time.time()),
        },
        is_verified=None,
    )
