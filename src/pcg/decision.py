from __future__ import annotations


def choose_action(risk: float, checker_valid: bool, threshold: float) -> str:
    if not checker_valid:
        return "refuse"
    if risk <= threshold:
        return "answer"
    return "refuse"
