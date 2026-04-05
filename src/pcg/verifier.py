from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(text)))


def jaccard(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


class Verifier:
    def best_choice(self, answer_text: str, choices: list[str]) -> tuple[int, str, float]:
        if not choices:
            return -1, "", 0.0

        best_idx = -1
        best_choice = ""
        best_score = -1.0

        ans = normalize_text(answer_text)

        # try explicit option letter patterns first
        explicit_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        tokens = ans.split()
        for t in tokens:
            if t in explicit_map and explicit_map[t] < len(choices):
                idx = explicit_map[t]
                return idx, choices[idx], 0.95

        # compare answer against each choice
        for idx, choice in enumerate(choices):
            c = normalize_text(choice)

            score = 0.55 * seq_ratio(ans, c) + 0.45 * jaccard(ans, c)

            # bonus if exact choice substring appears
            if c and c in ans:
                score += 0.15

            if score > best_score:
                best_score = score
                best_idx = idx
                best_choice = choice

        best_score = max(0.0, min(1.0, best_score))
        return best_idx, best_choice, best_score

    def score_risk(self, answer_text: str, choices: list[str]) -> tuple[float, int, str, float]:
        idx, choice, score = self.best_choice(answer_text, choices)
        risk = 1.0 - score
        return risk, idx, choice, score
