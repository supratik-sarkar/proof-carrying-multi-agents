from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jaccard_tokens(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_certificate(claim: str, evidence_texts: list[str], meta: dict[str, Any]) -> dict[str, Any]:
    normalized_claim = normalize_text(claim)

    evidence = []
    support_scores = []

    for idx, txt in enumerate(evidence_texts):
        txt_norm = normalize_text(txt)
        txt_hash = sha256_text(txt_norm)
        score = jaccard_tokens(normalized_claim, txt_norm)
        support_scores.append(score)

        evidence.append(
            {
                "eid": f"ev_{idx}",
                "text": txt,
                "normalized_text": txt_norm,
                "text_hash": txt_hash,
                "token_count": len(tokenize(txt_norm)),
                "support_score": score,
            }
        )

    # keep the top-2 support items as a simple "minimal support" proxy
    ranked = sorted(
        enumerate(support_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    minimal_support_ids = [f"ev_{i}" for i, _ in ranked[: min(2, len(ranked))]]

    payload = {
        "claim": claim,
        "normalized_claim": normalized_claim,
        "evidence": evidence,
        "minimal_support_ids": minimal_support_ids,
        "meta": meta,
    }

    payload_string = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload["certificate_hash"] = sha256_text(payload_string)

    return payload
