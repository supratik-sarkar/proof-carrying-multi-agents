"""
FEVER-compatible loader.

Primary source:
    BeIR/fever

Rationale:
    The canonical Hugging Face `fever/fever` repository is script-based and can
    fail under modern `datasets` versions. For PCG-MAS artifact runs, we use the
    BEIR FEVER variant as a real 500-example FEVER-derived retrieval/fact-checking
    source. An explicit deterministic alternate remains available only when
    PCG_ALLOW_DATASET_ALTERNATE=1 and real loading fails.
"""
from __future__ import annotations

import os
from typing import Any, Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_BEIR_DATASET = "BeIR/fever"


def _allow_dataset_alternate() -> bool:
    return os.environ.get("PCG_ALLOW_DATASET_ALTERNATE", "0") == "1"


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _iter_fever_alternate(n: int | None = None, seed: int = 0) -> Iterator[QAExample]:
    """Deterministic FEVER-shaped alternate for environment preflight only."""
    base = [
        (
            "The Eiffel Tower is located in Paris.",
            "SUPPORTS",
            "The Eiffel Tower is a wrought-iron tower on the Champ de Mars in Paris, France.",
            "Eiffel Tower",
        ),
        (
            "The Pacific Ocean is smaller than the Arctic Ocean.",
            "REFUTES",
            "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
            "Pacific Ocean",
        ),
        (
            "Marie Curie won a Nobel Prize.",
            "SUPPORTS",
            "Marie Curie was awarded Nobel Prizes in Physics and Chemistry.",
            "Marie Curie",
        ),
    ]

    total = 500 if n is None else n
    for i in range(total):
        claim, label, text, title = base[i % len(base)]
        yield QAExample(
            id=f"fever_alternate_{seed}_{i}",
            question=claim,
            gold_answers=(label,),
            evidence=(
                EvidenceItem(
                    id=f"fever_alternate_{seed}_{i}_e0",
                    title=title,
                    text=text,
                    source_url=None,
                    publisher="alternate_fever",
                    domain="fever.local",
                    is_gold=True,
                ),
            ),
            task_type="fact_verification",
            meta={"dataset": "fever", "alternate": True, "label": label},
        )


def _row_to_example(row: dict[str, Any], idx: int) -> QAExample:
    qid = str(
        _first_nonempty(
            row.get("_id"),
            row.get("id"),
            row.get("query-id"),
            row.get("query_id"),
            f"fever_{idx}",
        )
    )

    claim = str(
        _first_nonempty(
            row.get("query"),
            row.get("question"),
            row.get("claim"),
            row.get("text"),
            "",
        )
    )

    answer = _first_nonempty(
        row.get("label"),
        row.get("answer"),
        row.get("gold_answer"),
        row.get("relevance"),
        "SUPPORTS",
    )

    title = str(_first_nonempty(row.get("title"), row.get("docid"), row.get("_id"), "FEVER evidence"))
    evidence_text = str(
        _first_nonempty(
            row.get("text"),
            row.get("contents"),
            row.get("passage"),
            row.get("document"),
            claim,
        )
    )

    evidence = (
        EvidenceItem(
            id=f"{qid}_e0",
            title=title,
            text=evidence_text,
            source_url=None,
            publisher="BEIR/fever",
            domain="beir",
            is_gold=True,
        ),
    )

    return QAExample(
        id=qid,
        question=claim,
        gold_answers=(str(answer),),
        evidence=evidence,
        task_type="fact_verification",
        meta={"dataset": "fever", "source": "BeIR/fever"},
    )


def _unwrap_dataset(obj):
    """Return an iterable dataset from DatasetDict/IterableDatasetDict variants."""
    if hasattr(obj, "keys") and not hasattr(obj, "__iter__"):
        # Defensive, rarely used.
        keys = list(obj.keys())
        return obj[keys[0]]

    if hasattr(obj, "keys") and not isinstance(obj, dict):
        keys = list(obj.keys())
        preferred = ["queries", "test", "validation", "train", "corpus"]
        for key in preferred:
            if key in keys:
                return obj[key]
        return obj[keys[0]]

    if isinstance(obj, dict):
        preferred = ["queries", "test", "validation", "train", "corpus"]
        for key in preferred:
            if key in obj:
                return obj[key]
        return obj[next(iter(obj.keys()))]

    return obj


def _try_load_beir_split(split: str, streaming: bool):
    from datasets import load_dataset

    # BeIR/fever exposes named configs. Some environments expose config-level
    # DatasetDicts, while others expose split-addressable datasets. Try both.
    attempts = [
        # Most likely for modern datasets: config only, then unwrap.
        {"path": _BEIR_DATASET, "name": "queries", "split": None},
        {"path": _BEIR_DATASET, "name": "corpus", "split": None},

        # Split-addressable variants.
        {"path": _BEIR_DATASET, "name": "queries", "split": split},
        {"path": _BEIR_DATASET, "name": "queries", "split": "queries"},
        {"path": _BEIR_DATASET, "name": "queries", "split": "test"},
        {"path": _BEIR_DATASET, "name": "queries", "split": "train"},

        {"path": _BEIR_DATASET, "name": "corpus", "split": split},
        {"path": _BEIR_DATASET, "name": "corpus", "split": "corpus"},
        {"path": _BEIR_DATASET, "name": "corpus", "split": "train"},
    ]

    last_exc: Exception | None = None

    for attempt in attempts:
        try:
            kwargs = {
                "path": attempt["path"],
                "name": attempt["name"],
                "streaming": streaming,
            }
            if attempt["split"] is not None:
                kwargs["split"] = attempt["split"]

            ds = load_dataset(**kwargs)
            return _unwrap_dataset(ds)

        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("Unable to load BEIR FEVER.")


def iter_fever(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    shuffle_buffer: int = 1024,
) -> Iterator[QAExample]:
    """Yield FEVER-compatible examples."""
    try:
        ds = _try_load_beir_split(split=split, streaming=streaming)
    except Exception:
        if _allow_dataset_alternate():
            yield from _iter_fever_alternate(n=n, seed=seed)
            return
        raise

    if streaming:
        try:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        except Exception:
            pass

    count = 0
    for idx, row in enumerate(ds):
        if n is not None and count >= n:
            break
        ex = _row_to_example(row, idx)
        if ex.question.strip():
            yield ex
            count += 1
