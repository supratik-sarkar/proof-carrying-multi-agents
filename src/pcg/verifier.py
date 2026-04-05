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
    def score_answer(self, answer_text: str, choices: list[str]) -> dict:
        ans = normalize_text(answer_text)

        if not choices:
            return {
                "pred_idx": -1,
                "pred_choice": "",
                "score": 0.0,
                "risk": 1.0,
            }

        # explicit numeric / letter detection
        num_match = re.search(r"\b([1-4])\b", ans)
        if num_match:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(choices):
                return {
                    "pred_idx": idx,
                    "pred_choice": choices[idx],
                    "score": 0.95,
                    "risk": 0.05,
                }

        letter_match = re.search(r"\b([abcd])\b", ans)
        if letter_match:
            idx = {"a": 0, "b": 1, "c": 2, "d": 3}[letter_match.group(1)]
            if 0 <= idx < len(choices):
                return {
                    "pred_idx": idx,
                    "pred_choice": choices[idx],
                    "score": 0.93,
                    "risk": 0.07,
                }

        best_idx = -1
        best_choice = ""
        best_score = -1.0

        for idx, choice in enumerate(choices):
            c = normalize_text(choice)
            score = 0.6 * seq_ratio(ans, c) + 0.4 * jaccard(ans, c)

            if c and c in ans:
                score += 0.15

            if score > best_score:
                best_score = score
                best_idx = idx
                best_choice = choice

        best_score = max(0.0, min(1.0, best_score))
        return {
            "pred_idx": best_idx,
            "pred_choice": best_choice,
            "score": best_score,
            "risk": 1.0 - best_score,
        }
