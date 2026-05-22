"""
WebLINX loader.

WebLINX schemas differ by config ("reranking", "chat") and library version.
This loader robustly extracts an instruction/action target and compact evidence
from row fields without assuming a single schema.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "McGill-NLP/WebLINX"
_DATASET_CONFIG = "reranking"


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, str):
            s = str(value).strip()
            if s and s.lower() != "none":
                return s
    return ""


def _deep_find(obj: Any, keys: set[str]) -> Any:
    """Depth-first search for the first non-empty value whose key matches."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys and value not in (None, "", [], {}):
                return value
        for value in obj.values():
            found = _deep_find(value, keys)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _deep_find(value, keys)
            if found not in (None, "", [], {}):
                return found
    return None


def _json_preview(obj: Any, max_chars: int = 1200) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(obj)
    text = " ".join(text.split())
    return text[:max_chars]


def _extract_question(row: dict[str, Any]) -> str:
    direct = _first_nonempty(
        row.get("utterance"),
        row.get("query"),
        row.get("question"),
        row.get("instruction"),
        row.get("intent"),
        row.get("goal"),
        row.get("user_intent"),
        row.get("turn"),
    )
    if direct:
        return direct

    deep = _deep_find(
        row,
        {
            "utterance",
            "query",
            "question",
            "instruction",
            "intent",
            "goal",
            "user_intent",
            "user_utterance",
            "task",
        },
    )
    if deep not in (None, "", [], {}):
        return str(deep).strip()

    # Reranking rows may be candidate-oriented and not have a natural question.
    demo = _first_nonempty(row.get("demo_name"), row.get("demo_id"), row.get("session_id"))
    turn = _first_nonempty(row.get("turn_index"), row.get("turn_id"))
    url = _first_nonempty(row.get("url"), _deep_find(row, {"url", "page_url"}))
    if demo or turn or url:
        return f"Select the correct web action for demo={demo or 'unknown'}, turn={turn or 'unknown'}, url={url or 'unknown'}."

    return "Select the correct next web action from the available WebLINX context."


def _extract_gold_answers(row: dict[str, Any]) -> tuple[str, ...]:
    candidates = [
        row.get("target"),
        row.get("gold"),
        row.get("answer"),
        row.get("label"),
        row.get("action"),
        row.get("target_action"),
        row.get("positive"),
        row.get("correct"),
    ]

    deep = _deep_find(
        row,
        {"target", "gold", "answer", "label", "action", "target_action", "positive", "correct"},
    )
    candidates.append(deep)

    out: list[str] = []
    for item in candidates:
        if item in (None, "", [], {}):
            continue
        if isinstance(item, list):
            out.extend(str(x) for x in item if x not in (None, "", [], {}))
        elif isinstance(item, dict):
            out.append(_json_preview(item, max_chars=500))
        else:
            out.append(str(item))

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for x in out:
        x = x.strip()
        if x and x not in seen and x.lower() != "none":
            deduped.append(x)
            seen.add(x)

    return tuple(deduped) if deduped else ("web_action_target_unavailable",)


def _extract_evidence_text(row: dict[str, Any]) -> str:
    pieces: list[str] = []

    for key in [
        "url",
        "page_url",
        "title",
        "html",
        "dom",
        "viewport",
        "screenshot",
        "candidates",
        "candidate",
        "query",
        "utterance",
        "context",
        "history",
        "turns",
    ]:
        value = row.get(key)
        if value not in (None, "", [], {}):
            pieces.append(f"{key}: {_json_preview(value, max_chars=900)}")

    if not pieces:
        pieces.append("raw_row: " + _json_preview(row, max_chars=1500))

    return "\n".join(pieces)


def _row_to_example(row: dict[str, Any], idx: int) -> QAExample:
    qid = _first_nonempty(
        row.get("id"),
        row.get("uid"),
        row.get("turn_id"),
        row.get("demo_name"),
        row.get("demo_id"),
        f"weblinx_{idx}",
    )

    question = _extract_question(row)
    gold_answers = _extract_gold_answers(row)
    evidence_text = _extract_evidence_text(row)

    url = _first_nonempty(row.get("url"), row.get("page_url"), _deep_find(row, {"url", "page_url"}))
    demo_name = _first_nonempty(row.get("demo_name"), row.get("demo_id"), row.get("session_id"))

    evidence = (
        EvidenceItem(
            id=f"{qid}_web_context",
            title=f"WebLINX context {demo_name or idx}",
            text=evidence_text,
            source_url=url or None,
            publisher="WebLINX",
            domain="weblinx",
            is_gold=True,
        ),
    )

    return QAExample(
        id=str(qid),
        question=question,
        gold_answers=gold_answers,
        evidence=evidence,
        task_type="web_action",
        meta={
            "dataset": "weblinx",
            "demo_name": row.get("demo_name") or row.get("demo_id"),
            "turn_index": row.get("turn_index") or row.get("turn_id"),
            "intent": row.get("intent") or row.get("user_intent"),
            "url": url or None,
            "config": _DATASET_CONFIG,
        },
    )


def iter_weblinx(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    config: str = _DATASET_CONFIG,
) -> Iterator[QAExample]:
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
