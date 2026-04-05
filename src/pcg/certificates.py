from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jaccard(a: str, b: str) -> float:
    sa = set(tokens(a))
    sb = set(tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_certificate(claim: str, evidence_texts: list[str], meta: dict[str, Any]) -> dict[str, Any]:
    norm_claim = normalize_text(claim)

    evidence = []
    support_scores = []
    lengths = []

    for idx, txt in enumerate(evidence_texts):
        norm_txt = normalize_text(txt)
        score = jaccard(norm_claim, norm_txt)
        support_scores.append(score)
        lengths.append(len(tokens(norm_txt)))

        evidence.append(
            {
                "eid": f"ev_{idx}",
                "raw_text": txt,
                "normalized_text": norm_txt,
                "text_hash": sha256_text(norm_txt),
                "support_score": score,
                "token_count": len(tokens(norm_txt)),
            }
        )

    ranked = sorted(
        [(ev["eid"], ev["support_score"]) for ev in evidence],
        key=lambda x: x[1],
        reverse=True,
    )

    top_support = ranked[: min(3, len(ranked))]
    minimal_support_ids = [x[0] for x in top_support]

    certificate = {
        "claim": claim,
        "normalized_claim": norm_claim,
        "evidence": evidence,
        "minimal_support_ids": minimal_support_ids,
        "support_summary": {
            "max_support": max(support_scores) if support_scores else 0.0,
            "mean_support": sum(support_scores) / len(support_scores) if support_scores else 0.0,
            "mean_evidence_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        },
        "meta": meta,
    }

    certificate_payload = json.dumps(certificate, sort_keys=True, ensure_ascii=False)
    certificate["certificate_hash"] = sha256_text(certificate_payload)
    return certificate
