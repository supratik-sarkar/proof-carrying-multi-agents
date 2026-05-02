"""
Task metrics used across experiments.

We keep these small and dependency-free so that the experiment scripts can
be debugged without model downloads.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence


# -----------------------------------------------------------------------------
# Text normalization (HotpotQA / SQuAD convention)
# -----------------------------------------------------------------------------


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)


def _normalize_answer(s: str) -> str:
    """Lowercase, strip articles/punctuation/extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str | Iterable[str]) -> float:
    """EM = 1 if normalized pred matches any normalized gold, else 0."""
    npr = _normalize_answer(pred)
    if isinstance(gold, str):
        return 1.0 if npr == _normalize_answer(gold) else 0.0
    return 1.0 if any(npr == _normalize_answer(g) for g in gold) else 0.0


def f1_score(pred: str, gold: str | Iterable[str]) -> float:
    """Token-level F1 (HotpotQA convention).

    Returns the MAX F1 over reference answers when `gold` is a list.
    """
    golds = [gold] if isinstance(gold, str) else list(gold)
    best = 0.0
    pred_tokens = _normalize_answer(pred).split()
    if not pred_tokens:
        # Empty prediction: F1 = 1 iff gold is also empty
        return 1.0 if any(not _normalize_answer(g).split() for g in golds) else 0.0
    for g in golds:
        gold_tokens = _normalize_answer(g).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f = 2 * precision * recall / (precision + recall)
        best = max(best, f)
    return best


def success_rate(successes: Sequence[bool | int | float]) -> float:
    """Simple mean of 0/1 outcomes. Used for agent task success."""
    if not successes:
        return 0.0
    return float(sum(float(s) for s in successes)) / len(successes)
