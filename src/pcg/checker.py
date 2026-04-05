from __future__ import annotations

import json

from src.pcg.certificates import normalize_text, sha256_text, jaccard


def check_certificate(cert: dict) -> tuple[bool, list[str], dict]:
    reasons = []
    diagnostics = {
        "hash_pass_rate": 0.0,
        "support_recomputed_mean": 0.0,
        "support_recomputed_max": 0.0,
        "minimal_support_count": 0,
    }

    if not cert.get("claim"):
        reasons.append("missing_claim")
    if not cert.get("normalized_claim"):
        reasons.append("missing_normalized_claim")

    evidence = cert.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) == 0:
        reasons.append("missing_evidence")
        return False, reasons, diagnostics

    recomputed_scores = []
    hash_ok = 0

    for ev in evidence:
        eid = ev.get("eid", "unknown")
        raw = ev.get("raw_text")
        norm = ev.get("normalized_text")
        h = ev.get("text_hash")

        if raw is None:
            reasons.append(f"missing_raw_text_{eid}")
            continue
        if norm is None:
            reasons.append(f"missing_normalized_text_{eid}")
            continue
        if h is None:
            reasons.append(f"missing_text_hash_{eid}")
            continue

        recomputed_norm = normalize_text(raw)
        if recomputed_norm != norm:
            reasons.append(f"normalized_text_mismatch_{eid}")

        recomputed_hash = sha256_text(recomputed_norm)
        if recomputed_hash == h:
            hash_ok += 1
        else:
            reasons.append(f"hash_mismatch_{eid}")

        recomputed_scores.append(jaccard(cert["normalized_claim"], recomputed_norm))

    if len(evidence) > 0:
        diagnostics["hash_pass_rate"] = hash_ok / len(evidence)
    if recomputed_scores:
        diagnostics["support_recomputed_mean"] = sum(recomputed_scores) / len(recomputed_scores)
        diagnostics["support_recomputed_max"] = max(recomputed_scores)

    minimal_support_ids = cert.get("minimal_support_ids", [])
    diagnostics["minimal_support_count"] = len(minimal_support_ids)

    valid_ids = {ev.get("eid") for ev in evidence}
    for eid in minimal_support_ids:
        if eid not in valid_ids:
            reasons.append(f"missing_minimal_support_id_{eid}")

    cert_hash = cert.get("certificate_hash")
    if cert_hash is None:
        reasons.append("missing_certificate_hash")
    else:
        tmp = dict(cert)
        tmp.pop("certificate_hash", None)
        payload = json.dumps(tmp, sort_keys=True, ensure_ascii=False)
        recomputed_cert_hash = sha256_text(payload)
        if recomputed_cert_hash != cert_hash:
            reasons.append("certificate_hash_mismatch")

    is_valid = len(reasons) == 0
    return is_valid, reasons, diagnostics
