from __future__ import annotations


VALID_MODES = {
    "pcg_full",
    "baseline_selective",
    "baseline_multiagent_no_cert",
    "baseline_lightweight_citation",
    "baseline_posthoc_verify",
}


def validate_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
