"""
2WikiMultihopQA loader.

The HF row schema can vary across versions: contexts/supporting facts may be
dict-backed or list-backed. This loader normalizes both forms into QAExample.
"""
from __future__ import annotations

from typing import Any, Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "voidful/2WikiMultihopQA"


def _safe_get(obj: Any, key: str | int, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    if isinstance(obj, list) and isinstance(key, int) and 0 <= key < len(obj):
        return obj[key]
    return default


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _normalize_supporting_facts(sf: Any) -> set[str]:
    """Return normalized supporting titles from dict/list variants."""
    titles: set[str] = set()

    if isinstance(sf, dict):
        raw_titles = sf.get("title", [])
        for title in _as_list(raw_titles):
            if title is not None:
                titles.add(str(title))
        return titles

    if isinstance(sf, list):
        for item in sf:
            if isinstance(item, dict):
                title = item.get("title") or item.get("context") or item.get("name")
                if title is not None:
                    titles.add(str(title))
            elif isinstance(item, list) and item:
                title = item[0]
                if title is not None:
                    titles.add(str(title))
            elif isinstance(item, str):
                titles.add(item)

    return titles


def _normalize_contexts(ctx: Any) -> list[tuple[str, str]]:
    """Return list of (title, text) pairs from dict/list context variants."""
    out: list[tuple[str, str]] = []

    if isinstance(ctx, dict):
        titles = _as_list(ctx.get("title", []))
        sentences = _as_list(ctx.get("sentences", ctx.get("sentence", [])))

        for i, title in enumerate(titles):
            sent_obj = sentences[i] if i < len(sentences) else ""
            if isinstance(sent_obj, list):
                text = " ".join(str(s) for s in sent_obj)
            else:
                text = str(sent_obj)
            out.append((str(title), text))
        return out

    if isinstance(ctx, list):
        for item in ctx:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or item.get("context") or "context"
                sent_obj = item.get("sentences") or item.get("sentence") or item.get("text") or ""
                if isinstance(sent_obj, list):
                    text = " ".join(str(s) for s in sent_obj)
                else:
                    text = str(sent_obj)
                out.append((str(title), text))

            elif isinstance(item, list):
                if len(item) >= 2:
                    title = str(item[0])
                    sent_obj = item[1]
                    if isinstance(sent_obj, list):
                        text = " ".join(str(s) for s in sent_obj)
                    else:
                        text = str(sent_obj)
                    out.append((title, text))
                elif len(item) == 1:
                    out.append(("context", str(item[0])))

            elif isinstance(item, str):
                out.append(("context", item))

    return out


def _row_to_example(row: dict[str, Any], idx: int) -> QAExample:
    qid = str(row.get("_id") or row.get("id") or f"twowiki_{idx}")
    question = str(row.get("question") or row.get("query") or "")
    answer = row.get("answer", "")

    contexts = _normalize_contexts(row.get("context") or row.get("contexts"))
    gold_titles = _normalize_supporting_facts(
        row.get("supporting_facts") or row.get("supportingfacts") or row.get("evidence")
    )

    evidence: list[EvidenceItem] = []
    for k, (title, text) in enumerate(contexts):
        evidence.append(
            EvidenceItem(
                id=f"{qid}_ctx{k}",
                title=title,
                text=text,
                source_url=None,
                publisher="2WikiMultihopQA",
                domain="2wikimultihopqa",
                is_gold=(title in gold_titles) if gold_titles else False,
            )
        )

    if not evidence:
        evidence.append(
            EvidenceItem(
                id=f"{qid}_ctx0",
                title="context",
                text=str(row),
                source_url=None,
                publisher="2WikiMultihopQA",
                domain="2wikimultihopqa",
                is_gold=False,
            )
        )

    return QAExample(
        id=qid,
        question=question,
        gold_answers=(str(answer),) if answer is not None else tuple(),
        evidence=tuple(evidence),
        task_type="qa",
        meta={
            "dataset": "twowiki",
            "type": row.get("type") or row.get("hop_type"),
            "supporting_titles": sorted(gold_titles),
        },
    )


def iter_twowiki(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
) -> Iterator[QAExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "huggingface `datasets` is not installed. Install via `pip install datasets`."
        ) from exc

    ds = load_dataset(_DATASET_NAME, split=split, streaming=streaming)

    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=1024)

    count = 0
    for idx, row in enumerate(ds):
        if n is not None and count >= n:
            break
        yield _row_to_example(row, idx)
        count += 1
