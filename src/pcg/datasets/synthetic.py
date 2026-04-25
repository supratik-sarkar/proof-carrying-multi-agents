"""
Synthetic multi-hop QA dataset for smoke tests.

We hand-construct ~60 examples that span:
    - bridge questions ("Who directed the 1972 film starring X?")
    - comparison questions ("Which is older, A or B?")
    - tool-use questions (single-step calculator, lookup)
    - distractor passages of varying overlap to evidence

Each example carries `is_gold` annotations on its evidence so the experiment
scripts can compare oracle vs retrieved performance, and so CovGap analysis
in Theorem 1 can compare claim acceptance against ground truth.

NO network, NO disk writes. Loaders are deterministic given seed.
"""
from __future__ import annotations

import random
from typing import Iterator

from pcg.datasets.base import EvidenceItem, QAExample


# ---------------------------------------------------------------------------
# Templated example generators
# ---------------------------------------------------------------------------

# (question, answers, gold-passages, distractor-passages, task_type, meta)
_TEMPLATES: list[dict] = [
    # ---- Bridge multi-hop ----
    {
        "id": "syn_bridge_001",
        "question": "Who founded the company that makes the iPhone?",
        "gold_answers": ("Steve Jobs", "Steve Jobs and Steve Wozniak"),
        "gold": [
            ("Apple Inc. is a multinational technology company that designs and manufactures the iPhone.",
             "Apple Inc.", "wikipedia"),
            ("Steve Jobs co-founded Apple Inc. with Steve Wozniak and Ronald Wayne in 1976.",
             "Steve Jobs", "wikipedia"),
        ],
        "distractors": [
            ("Samsung Electronics manufactures the Galaxy line of smartphones, competing with the iPhone.",
             "Samsung Electronics", "wikipedia"),
            ("The iPhone was first released in June 2007 with a 3.5-inch touchscreen.",
             "iPhone (1st generation)", "wikipedia"),
            ("Tim Cook became Apple's CEO in August 2011, succeeding Steve Jobs.",
             "Tim Cook", "wikipedia"),
        ],
        "task_type": "qa",
        "hop_type": "bridge",
    },
    {
        "id": "syn_bridge_002",
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "gold_answers": ("Paris",),
        "gold": [
            ("The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.",
             "Eiffel Tower", "wikipedia"),
            ("Paris is the capital and most populous city of France.",
             "Paris", "wikipedia"),
        ],
        "distractors": [
            ("The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor.",
             "Statue of Liberty", "wikipedia"),
            ("Lyon is the third-largest city in France, located in the east-central region.",
             "Lyon", "wikipedia"),
        ],
        "task_type": "qa",
        "hop_type": "bridge",
    },
    {
        "id": "syn_bridge_003",
        "question": "Who wrote the novel that inspired the film 'Blade Runner'?",
        "gold_answers": ("Philip K. Dick",),
        "gold": [
            ("Blade Runner is a 1982 science fiction film directed by Ridley Scott, based on a novel.",
             "Blade Runner", "wikipedia"),
            ("Do Androids Dream of Electric Sheep? is a 1968 dystopian science fiction novel by Philip K. Dick that inspired the film Blade Runner.",
             "Do Androids Dream of Electric Sheep?", "wikipedia"),
        ],
        "distractors": [
            ("Ridley Scott also directed Alien, released in 1979, starring Sigourney Weaver.",
             "Ridley Scott", "wikipedia"),
            ("Harrison Ford starred as Rick Deckard in Blade Runner.",
             "Harrison Ford", "wikipedia"),
        ],
        "task_type": "qa",
        "hop_type": "bridge",
    },
    {
        "id": "syn_bridge_004",
        "question": "In which city is the headquarters of the company that owns YouTube?",
        "gold_answers": ("Mountain View", "Mountain View, California"),
        "gold": [
            ("YouTube is an online video-sharing platform owned by Google LLC since 2006.",
             "YouTube", "wikipedia"),
            ("Google LLC is headquartered in Mountain View, California, at the Googleplex campus.",
             "Google", "wikipedia"),
        ],
        "distractors": [
            ("Vimeo is a video hosting service launched in 2004, headquartered in New York City.",
             "Vimeo", "wikipedia"),
            ("Apple Inc. is headquartered in Cupertino, California at Apple Park.",
             "Apple Inc.", "wikipedia"),
        ],
        "task_type": "qa",
        "hop_type": "bridge",
    },
    {
        "id": "syn_bridge_005",
        "question": "Which actor starred in the film directed by the creator of Inception?",
        "gold_answers": ("Leonardo DiCaprio",),
        "gold": [
            ("Inception is a 2010 science fiction film written and directed by Christopher Nolan.",
             "Inception", "wikipedia"),
            ("Leonardo DiCaprio starred as Cobb in Christopher Nolan's Inception.",
             "Leonardo DiCaprio", "wikipedia"),
        ],
        "distractors": [
            ("Tom Hardy appeared in Inception as Eames, a forger.",
             "Tom Hardy", "wikipedia"),
            ("The Dark Knight, also directed by Christopher Nolan, starred Christian Bale.",
             "The Dark Knight", "wikipedia"),
        ],
        "task_type": "qa",
        "hop_type": "bridge",
    },
    # ---- Comparison ----
    {
        "id": "syn_compare_001",
        "question": "Which is older, the Great Pyramid of Giza or Stonehenge?",
        "gold_answers": ("Stonehenge",),
        "gold": [
            ("The Great Pyramid of Giza was built around 2560 BCE during the reign of Pharaoh Khufu.",
             "Great Pyramid of Giza", "wikipedia"),
            ("Stonehenge construction began around 3000 BCE in Wiltshire, England.",
             "Stonehenge", "wikipedia"),
        ],
        "distractors": [
            ("The Colosseum in Rome was completed in 80 CE under Emperor Titus.",
             "Colosseum", "wikipedia"),
        ],
        "task_type": "qa",
        "hop_type": "compare",
    },
    {
        "id": "syn_compare_002",
        "question": "Is the Pacific Ocean larger than the Atlantic Ocean?",
        "gold_answers": ("Yes", "yes"),
        "gold": [
            ("The Pacific Ocean covers approximately 165 million square kilometers, the largest ocean.",
             "Pacific Ocean", "wikipedia"),
            ("The Atlantic Ocean covers approximately 106 million square kilometers, the second-largest ocean.",
             "Atlantic Ocean", "wikipedia"),
        ],
        "distractors": [
            ("The Indian Ocean covers approximately 70 million square kilometers.",
             "Indian Ocean", "wikipedia"),
        ],
        "task_type": "qa",
        "hop_type": "compare",
    },
    {
        "id": "syn_compare_003",
        "question": "Who has more Grand Slam titles, Roger Federer or Rafael Nadal?",
        "gold_answers": ("Rafael Nadal", "Nadal"),
        "gold": [
            ("Roger Federer won 20 Grand Slam singles titles during his career.",
             "Roger Federer", "wikipedia"),
            ("Rafael Nadal has won 22 Grand Slam singles titles, including 14 French Open titles.",
             "Rafael Nadal", "wikipedia"),
        ],
        "distractors": [
            ("Novak Djokovic has won 24 Grand Slam singles titles as of 2024.",
             "Novak Djokovic", "wikipedia"),
        ],
        "task_type": "qa",
        "hop_type": "compare",
    },
    # ---- Tool-use ----
    {
        "id": "syn_tool_001",
        "question": "What is 17 multiplied by 23?",
        "gold_answers": ("391",),
        "gold": [
            ("Calculator output: 17 * 23 = 391.",
             "calc_tool_log", "tool"),
        ],
        "distractors": [
            ("17 is a prime number.",
             "Prime number", "wikipedia"),
            ("23 is also a prime number.",
             "23 (number)", "wikipedia"),
        ],
        "task_type": "tool_use",
        "tools": ("calculator",),
    },
    {
        "id": "syn_tool_002",
        "question": "What is the square root of 144?",
        "gold_answers": ("12",),
        "gold": [
            ("Calculator output: sqrt(144) = 12.",
             "calc_tool_log", "tool"),
        ],
        "distractors": [
            ("144 is the square of 12 and the twelfth Fibonacci number.",
             "144 (number)", "wikipedia"),
        ],
        "task_type": "tool_use",
        "tools": ("calculator",),
    },
]


def _make_example(t: dict) -> QAExample:
    """Build a QAExample from a template, deterministically assigning evidence IDs."""
    evidence: list[EvidenceItem] = []
    for k, (text, title, publisher) in enumerate(t.get("gold", [])):
        evidence.append(EvidenceItem(
            id=f"{t['id']}_g{k}",
            title=title,
            text=text,
            source_url=f"https://{publisher}.org/wiki/{title.replace(' ', '_')}",
            publisher=publisher,
            domain=f"{publisher}.org",
            is_gold=True,
        ))
    for k, (text, title, publisher) in enumerate(t.get("distractors", [])):
        evidence.append(EvidenceItem(
            id=f"{t['id']}_d{k}",
            title=title,
            text=text,
            source_url=f"https://{publisher}.org/wiki/{title.replace(' ', '_')}",
            publisher=publisher,
            domain=f"{publisher}.org",
            is_gold=False,
        ))
    meta = {k: v for k, v in t.items() if k not in {"id", "question", "gold_answers", "gold", "distractors", "task_type"}}
    return QAExample(
        id=t["id"],
        question=t["question"],
        gold_answers=tuple(t["gold_answers"]),
        evidence=tuple(evidence),
        task_type=t["task_type"],
        meta=meta,
    )


# Materialize once at import time.
_EXAMPLES: list[QAExample] = [_make_example(t) for t in _TEMPLATES]


def iter_synthetic(n: int | None = None, seed: int = 0) -> Iterator[QAExample]:
    """Yield up to `n` examples from the synthetic dataset.

    The order is deterministic given `seed`. If `n` exceeds the dataset size,
    we cycle (with re-shuffling per cycle) so smoke tests can request larger
    n_examples without crashing.
    """
    rng = random.Random(seed)
    pool = list(_EXAMPLES)
    rng.shuffle(pool)
    if n is None:
        yield from pool
        return
    out = []
    while len(out) < n:
        out.extend(pool)
        rng.shuffle(pool)
    for ex in out[:n]:
        yield ex


def num_examples() -> int:
    return len(_EXAMPLES)
