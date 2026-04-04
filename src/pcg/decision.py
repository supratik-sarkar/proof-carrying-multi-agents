from __future__ import annotations


def choose_action(risk: float, threshold: float) -> str:
    if risk > threshold:
        return "refuse"
    return "answer"
