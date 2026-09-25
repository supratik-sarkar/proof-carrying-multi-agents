"""
Task metrics used across experiments.

We keep these small and dependency-free so that the experiment scripts can
be debugged without model downloads.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, Any, Iterable, Sequence


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


# -----------------------------------------------------------------------------
# Validation Architectural Metrics (H_support / H_exec, S & V Decomposition, Shift Gate)
# -----------------------------------------------------------------------------


def compute_harm_decomposition(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Decomposes composite harm into H_support (citation/entailment) and H_exec (execution/tool policy)."""
    if not records:
        return {"H_support": 0.0, "H_exec": 0.0, "composite_harm": 0.0}

    n = len(records)
    h_supp = sum(1.0 for r in records if r.get("unsupported_claim", False) or r.get("entailment_fail", False)) / n
    h_exec = sum(1.0 for r in records if r.get("disallowed_tool", False) or r.get("delegation_breach", False)) / n
    h_comp = sum(1.0 for r in records if (r.get("unsupported_claim") or r.get("entailment_fail") or r.get("disallowed_tool") or r.get("delegation_breach"))) / n

    return {
        "H_support": round(h_supp, 4),
        "H_exec": round(h_exec, 4),
        "composite_harm": round(h_comp, 4)
    }


def compute_applicability_aware_harm(
    records: Sequence[Dict[str, Any]],
    *,
    grounding_applicable: bool = True,
    policy_applicable: bool = True,
) -> Dict[str, Any]:
    """Computes applicability-aware harm decomposition.
    Non-applicable harm channels evaluate to False (0).
    Zero denominator yields None (UNDEFINED), never 0.0000.
    """
    if not records:
        return {"H_support": None, "H_exec": None, "H_joint": None, "N": 0}

    n = len(records)
    supp_count = sum(1 for r in records if r.get("h_support", False)) if grounding_applicable else 0
    exec_count = sum(1 for r in records if r.get("h_exec", False)) if policy_applicable else 0

    joint_count = 0
    for r in records:
        s = bool(r.get("h_support", False)) if grounding_applicable else False
        e = bool(r.get("h_exec", False)) if policy_applicable else False
        if s or e:
            joint_count += 1

    return {
        "H_support": round(supp_count / n, 4) if grounding_applicable else 0.0,
        "H_exec": round(exec_count / n, 4) if policy_applicable else 0.0,
        "H_joint": round(joint_count / n, 4),
        "N": n,
        "numerator": joint_count,
        "denominator": n,
    }


def compute_sv_decomposition(
    harm_nocert: float,
    harm_pcg: float,
    accept_rate: float
) -> Dict[str, float]:
    """Computes S (selectivity harm avoided) and V (verification harm avoided on same answered set).

    S = harm_nocert * (1 - accept_rate)
    V = accept_rate * (harm_nocert - harm_pcg)
    """
    S = harm_nocert * (1.0 - accept_rate)
    V = accept_rate * (harm_nocert - harm_pcg)
    return {
        "S_selectivity_harm_avoided": round(S, 4),
        "V_verification_harm_avoided": round(V, 4),
        "total_harm_reduction": round(S + V, 4)
    }


def check_ucb_rho_gate(realized_rhos: Sequence[float], bar_rho: float, delta: float = 0.05, n_samples: int | None = None) -> Dict[str, Any]:
    """Evaluates UCB gate for co-failure parameter: hat_rho_UCB <= bar_rho + Delta."""
    if not realized_rhos:
        return {"hat_rho_ucb": 0.0, "gate_passed": True}

    n = n_samples if n_samples is not None else len(realized_rhos)
    mean_rho = float(sum(realized_rhos)) / len(realized_rhos)
    hoeffding_margin = math.sqrt(math.log(1.0 / delta) / (2 * max(n, 50)))
    hat_rho_ucb = min(1.0, mean_rho + hoeffding_margin)

    passed = hat_rho_ucb <= (bar_rho + delta)
    return {
        "mean_rho": round(mean_rho, 4),
        "hat_rho_ucb": round(hat_rho_ucb, 4),
        "bar_rho_target": round(bar_rho, 4),
        "gate_passed": passed
    }
