from __future__ import annotations


class Verifier:
    def score_risk(self, answer_text: str, choices: list[str]) -> float:
        if not answer_text:
            return 1.0

        ans = answer_text.lower()

        for c in choices:
            if c and c.lower() in ans:
                return 0.15

        return 0.55
