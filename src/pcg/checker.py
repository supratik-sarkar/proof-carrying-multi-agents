from __future__ import annotations


def check_certificate(cert: dict) -> tuple[bool, list[str]]:
    reasons = []

    if not cert.get("claim"):
        reasons.append("missing_claim")

    if "evidence" not in cert or not isinstance(cert["evidence"], list):
        reasons.append("missing_evidence")

    for ev in cert.get("evidence", []):
        if "text" not in ev:
            reasons.append("missing_evidence_text")
        if "text_hash" not in ev:
            reasons.append("missing_evidence_hash")

    return (len(reasons) == 0, reasons)
