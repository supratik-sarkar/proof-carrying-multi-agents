from __future__ import annotations

import hashlib
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_certificate(claim: str, evidence_texts: list[str], meta: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for idx, txt in enumerate(evidence_texts):
        evidence.append(
            {
                "eid": f"ev_{idx}",
                "text": txt,
                "text_hash": sha256_text(txt),
            }
        )

    return {
        "claim": claim,
        "evidence": evidence,
        "meta": meta,
    }
