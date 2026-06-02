"""Stress-test the user's input against a small panel of attacks.

We deliberately reproduce a few canonical attacks locally rather than importing
from src/pcg/agents/attacker.py (which depends on the full pcg package and its
dataset/typing primitives). The point of the demo is to show PCG-MAS handling
adversarial perturbations of evidence; that doesn't require the full benchmark
attack suite — just clean text mutations the user can see.
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from typing import Callable

from .sources import ResolvedInput, Evidence


ATTACK_DESCRIPTIONS = {
    "evidence_shuffle":      "Reorder evidence chunks (should not affect a robust system).",
    "distractor_insertion":  "Insert an irrelevant evidence chunk into the set.",
    "evidence_deletion":     "Drop a random evidence chunk to simulate retrieval loss.",
    "contradiction_inject":  "Inject a contradicting evidence chunk.",
    "answer_evidence_swap":  "Replace an evidence chunk's content with unrelated text.",
    "claim_typo":            "Add typos to the claim (should not change semantics).",
    "publisher_obfuscate":   "Strip publisher metadata from all evidence.",
    "punctuation_strip":     "Remove punctuation from claim and evidence.",
}


@dataclass
class AttackedInput:
    name: str
    description: str
    resolved: ResolvedInput


def _clone(resolved: ResolvedInput) -> ResolvedInput:
    return ResolvedInput(
        claim=resolved.claim,
        evidence=[Evidence(**ev.__dict__) for ev in resolved.evidence],
        source_kind=resolved.source_kind,
        source_label=resolved.source_label,
    )


def _evidence_shuffle(r: ResolvedInput, seed: int) -> ResolvedInput:
    out = _clone(r)
    rng = random.Random(seed)
    rng.shuffle(out.evidence)
    for i, ev in enumerate(out.evidence):
        ev.rank = i
    out.source_label += " [shuffled]"
    return out


def _distractor_insertion(r: ResolvedInput, seed: int) -> ResolvedInput:
    out = _clone(r)
    distractor = Evidence(
        text="The Eiffel Tower in Paris is 330 metres tall and was completed in 1889.",
        title="(distractor)", publisher="adversary", rank=len(out.evidence),
    )
    rng = random.Random(seed)
    pos = rng.randint(0, len(out.evidence))
    out.evidence.insert(pos, distractor)
    for i, ev in enumerate(out.evidence):
        ev.rank = i
    out.source_label += " [+distractor]"
    return out


def _evidence_deletion(r: ResolvedInput, seed: int) -> ResolvedInput:
    out = _clone(r)
    if len(out.evidence) > 1:
        rng = random.Random(seed)
        idx = rng.randrange(len(out.evidence))
        out.evidence.pop(idx)
        for i, ev in enumerate(out.evidence):
            ev.rank = i
    out.source_label += " [evidence dropped]"
    return out


def _contradiction_inject(r: ResolvedInput, seed: int) -> ResolvedInput:
    out = _clone(r)
    if out.evidence:
        first = out.evidence[0].text
        contradiction = Evidence(
            text=f"Contrary to the previous claim, the opposite is true: {first[:120]} — this is false.",
            title="(contradiction)", publisher="adversary",
            rank=len(out.evidence),
        )
        out.evidence.append(contradiction)
        for i, ev in enumerate(out.evidence):
            ev.rank = i
    out.source_label += " [+contradiction]"
    return out


def _answer_evidence_swap(r: ResolvedInput, seed: int) -> ResolvedInput:
    out = _clone(r)
    if out.evidence:
        rng = random.Random(seed)
        idx = rng.randrange(len(out.evidence))
        out.evidence[idx].text = (
            "The capital of Liechtenstein is Vaduz, which has a population of "
            "approximately 5,500 residents as of recent estimates."
        )
        out.evidence[idx].title = "(swapped content)"
    out.source_label += " [evidence swapped]"
    return out


def _claim_typo(r: ResolvedInput, seed: int) -> ResolvedInput:
    out = _clone(r)
    rng = random.Random(seed)
    chars = list(out.claim)
    n_typos = max(1, len(chars) // 30)
    for _ in range(n_typos):
        if len(chars) < 2:
            break
        i = rng.randrange(1, len(chars) - 1)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    out.claim = "".join(chars)
    out.source_label += " [claim typos]"
    return out


def _publisher_obfuscate(r: ResolvedInput, seed: int) -> ResolvedInput:
    out = _clone(r)
    for ev in out.evidence:
        ev.publisher = ""
        ev.title = ""
        ev.url = ""
    out.source_label += " [publisher stripped]"
    return out


def _punctuation_strip(r: ResolvedInput, seed: int) -> ResolvedInput:
    out = _clone(r)
    out.claim = re.sub(r"[^\w\s]", "", out.claim)
    for ev in out.evidence:
        ev.text = re.sub(r"[^\w\s]", "", ev.text)
    out.source_label += " [punctuation stripped]"
    return out


ATTACK_REGISTRY: dict[str, Callable[[ResolvedInput, int], ResolvedInput]] = {
    "evidence_shuffle":     _evidence_shuffle,
    "distractor_insertion": _distractor_insertion,
    "evidence_deletion":    _evidence_deletion,
    "contradiction_inject": _contradiction_inject,
    "answer_evidence_swap": _answer_evidence_swap,
    "claim_typo":           _claim_typo,
    "publisher_obfuscate":  _publisher_obfuscate,
    "punctuation_strip":    _punctuation_strip,
}


def all_attacks(resolved: ResolvedInput, seed: int = 0) -> list[AttackedInput]:
    return [
        AttackedInput(
            name=name,
            description=ATTACK_DESCRIPTIONS[name],
            resolved=fn(resolved, seed),
        )
        for name, fn in ATTACK_REGISTRY.items()
    ]
