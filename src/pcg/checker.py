from __future__ import annotations

import json

from src.pcg.certificates import normalize_text, sha256_text, jaccard_tokens


def check_certificate(cert: dict) -> tuple[bool, list[str]]:
    reasons = []

    if not cert.get("claim"):
        reasons.append("missing_claim")

    if not cert.get("normalized_claim"):
        reasons.append("missing_normalized_claim")

    evidence = cert.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) == 0:
        reasons.append("missing_evidence")
        return False, reasons

    # recompute per-evidence hashes and verify structure
    recomputed_supports = {}
    for ev in evidence:
        if "eid" not in ev:
            reasons.append("missing_evidence_id")
            continue
        if "text" not in ev:
            reasons.append("missing_evidence_text")
            continue
        if "normalized_text" not in ev:
            reasons.append(f"missing_normalized_text_{ev.get('eid', 'unknown')}")
            continue
        if "text_hash" not in ev:
            reasons.append(f"missing_text_hash_{ev.get('eid', 'unknown')}")
            continue

        recomputed_norm = normalize_text(ev["text"])
        if recomputed_norm != ev["normalized_text"]:
            reasons.append(f"normalized_text_mismatch_{ev['eid']}")

        recomputed_hash = sha256_text(recomputed_norm)
        if recomputed_hash != ev["text_hash"]:
            reasons.append(f"hash_mismatch_{ev['eid']}")

        recomputed_supports[ev["eid"]] = jaccard_tokens(cert["normalized_claim"], recomputed_norm)

    # verify minimal support IDs exist
    minimal_support_ids = cert.get("minimal_support_ids", [])
    if not isinstance(minimal_support_ids, list):
        reasons.append("bad_minimal_support_ids_type")
    else:
        for eid in minimal_support_ids:
            if eid not in recomputed_supports:
                reasons.append(f"missing_minimal_support_eid_{eid}")

    # verify certificate hash
    certificate_hash = cert.get("certificate_hash")
    if not certificate_hash:
        reasons.append("missing_certificate_hash")
    else:
        tmp = dict(cert)
        tmp.pop("certificate_hash", None)
        recomputed_payload_string = json.dumps(tmp, sort_keys=True, ensure_ascii=False)
        recomputed_cert_hash = sha256_text(recomputed_payload_string)
        if recomputed_cert_hash != certificate_hash:
            reasons.append("certificate_hash_mismatch")

    return (len(reasons) == 0), reasons
