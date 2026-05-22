"""
PubMedQA loader.

The commonly used HF PubMedQA configuration exposes only a train split in some
environments. We map validation/dev/test requests to train for pipeline checks.
"""
from __future__ import annotations

from typing import Any, Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "qiaojin/PubMedQA"
_DATASET_CONFIG = "pqa_labeled"


def _normalize_split(split: str) -> str:
    if split in {"validation", "valid", "dev", "test"}:
        return "train"
    return split


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _context_to_text(context: Any) -> str:
    if isinstance(context, dict):
        labels = _as_list(context.get("labels", []))
        contexts = _as_list(context.get("contexts", []))
        if contexts:
            chunks = []
            for i, c in enumerate(contexts):
                label = labels[i] if i < len(labels) else ""
                prefix = f"{label}: " if label else ""
                chunks.append(prefix + str(c))
            return " ".join(chunks)
        return " ".join(str(v) for v in context.values())

    if isinstance(context, list):
        return " ".join(str(x) for x in context)

    return str(context or "")


def _row_to_example(row: dict[str, Any], idx: int) -> QAExample:
    qid = str(row.get("pubid") or row.get("id") or f"pubmedqa_{idx}")
    question = str(row.get("question") or "")
    answer = row.get("final_decision") or row.get("answer") or row.get("label") or ""
    long_answer = row.get("long_answer") or row.get("LONG_ANSWER") or ""

    context_text = _context_to_text(row.get("context") or row.get("CONTEXTS") or row.get("contexts"))

    evidence_text = context_text
    if long_answer:
        evidence_text = f"{context_text}\n\nLong answer: {long_answer}".strip()

    evidence = (
        EvidenceItem(
            id=f"{qid}_abstract",
            title="PubMedQA abstract context",
            text=evidence_text,
            source_url=None,
            publisher="PubMedQA",
            domain="pubmedqa",
            is_gold=True,
        ),
    )

    return QAExample(
        id=qid,
        question=question,
        gold_answers=(str(answer),) if answer else tuple(),
        evidence=evidence,
        task_type="biomedical_qa",
        meta={
            "dataset": "pubmedqa",
            "final_decision": answer,
            "long_answer": long_answer,
        },
    )


def iter_pubmedqa(
    *,
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    config: str = _DATASET_CONFIG,
) -> Iterator[QAExample]:
    split = _normalize_split(split)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "huggingface `datasets` is not installed. Install via `pip install datasets`."
        ) from exc

    ds = load_dataset(_DATASET_NAME, config, split=split, streaming=streaming)

    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=1024)

    count = 0
    for idx, row in enumerate(ds):
        if n is not None and count >= n:
            break
        yield _row_to_example(row, idx)
        count += 1
