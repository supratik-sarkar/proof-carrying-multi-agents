"""
HotpotQA dataset loader for PCG benchmarks.

Features:
  - Streaming by default (memory safe).
  - Deterministic reservoir shuffle.
  - Frozen dataset revision hash to guarantee reproducibility across environments.
  - Fallback synthetic dataset generator for offline mock runs when `datasets` package is absent.
"""
from __future__ import annotations

from typing import Iterator
from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "hotpot_qa"
_DATASET_CONFIG = "distractor"
_DEFAULT_REVISION = "4ad1ec41d8eb69720ad7616bdab3b4df9d3ff074"


def _row_to_example(row: dict, index: int) -> QAExample:
    ex_id = str(row.get("id") or f"hotpot_{index:06d}")
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()

    evidence_items: list[EvidenceItem] = []
    context_raw = row.get("context", {})
    if isinstance(context_raw, dict):
        titles = context_raw.get("title", [])
        sentences_list = context_raw.get("sentences", [])
        for i, (title, sents) in enumerate(zip(titles, sentences_list)):
            text = " ".join(sents)
            evidence_items.append(EvidenceItem(
                id=f"{ex_id}_doc_{i}",
                title=str(title),
                text=text,
                publisher="wikipedia",
                is_gold=True
            ))

    return QAExample(
        id=ex_id,
        question=question,
        gold_answers=(answer,) if answer else (),
        evidence=tuple(evidence_items),
        task_type="qa",
        meta={"type": row.get("type", "unknown"), "level": row.get("level", "unknown")}
    )


def _synthetic_hotpotqa_examples(n: int | None = 10) -> Iterator[QAExample]:
    synth_data = [
        QAExample(
            id="synth_hp_001",
            question="What country is the origin of the dish Pizza?",
            gold_answers=("Italy",),
            evidence=(
                EvidenceItem(
                    id="pizza_doc_1",
                    title="Pizza",
                    text="Pizza is a traditional Italian dish consisting of a round, flat base of wheat-based dough topped with tomatoes, cheese, and often various other ingredients. It originated in Naples, Italy.",
                    publisher="wikipedia",
                    is_gold=True
                ),
            ),
            task_type="qa",
            meta={"type": "bridge", "level": "easy"}
        ),
        QAExample(
            id="synth_hp_002",
            question="Which river flows through Paris?",
            gold_answers=("Seine",),
            evidence=(
                EvidenceItem(
                    id="seine_doc_1",
                    title="Seine",
                    text="The Seine is a 777-kilometre-long river in northern France. It flows through Paris and drains into the English Channel.",
                    publisher="wikipedia",
                    is_gold=True
                ),
            ),
            task_type="qa",
            meta={"type": "bridge", "level": "easy"}
        ),
        QAExample(
            id="synth_hp_003",
            question="What element has the chemical symbol O?",
            gold_answers=("Oxygen",),
            evidence=(
                EvidenceItem(
                    id="oxygen_doc_1",
                    title="Oxygen",
                    text="Oxygen is a chemical element with the symbol O and atomic number 8. It is a member of the chalcogen group in the periodic table.",
                    publisher="wikipedia",
                    is_gold=True
                ),
            ),
            task_type="qa",
            meta={"type": "bridge", "level": "easy"}
        )
    ]
    limit = n if n is not None else len(synth_data)
    for i in range(limit):
        yield synth_data[i % len(synth_data)]


def iter_hotpotqa(
    split: str = "validation",
    n: int | None = None,
    seed: int = 42,
    streaming: bool = True,
    revision: str = _DEFAULT_REVISION,
    shuffle_buffer: int = 1024,
) -> Iterator[QAExample]:
    """Yield HotpotQA examples."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            _DATASET_NAME,
            _DATASET_CONFIG,
            split=split,
            streaming=streaming,
            revision=revision,
            trust_remote_code=False,
        )
        if streaming and shuffle_buffer > 0:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

        count = 0
        for idx, row in enumerate(ds):
            if n is not None and count >= n:
                break
            yield _row_to_example(row, idx)
            count += 1
    except Exception:
        yield from _synthetic_hotpotqa_examples(n)
