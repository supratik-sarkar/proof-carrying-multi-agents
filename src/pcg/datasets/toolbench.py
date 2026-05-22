"""
ToolBench loader.

Primary source:
    tuandunghcmut/toolbench-v1

This is a ToolBench/ToolLLaMA-style instruction tuning dataset. The loader
normalizes tool-use rows into QAExample objects with a user instruction,
tool/API evidence, and a target answer/path when available.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_PRIMARY_DATASET = "tuandunghcmut/toolbench-v1"


def _allow_dataset_alternate() -> bool:
    return os.environ.get("PCG_ALLOW_DATASET_ALTERNATE", "0") == "1"


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _json_preview(obj: Any, max_chars: int = 2500) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(obj)
    text = " ".join(text.split())
    return text[:max_chars]


def _deep_find(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and v not in (None, "", [], {}):
                return v
        for v in obj.values():
            found = _deep_find(v, keys)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _deep_find(v, keys)
            if found not in (None, "", [], {}):
                return found
    return None


def _extract_instruction(row: dict[str, Any]) -> str:
    direct = _first_nonempty(
        row.get("instruction"),
        row.get("query"),
        row.get("question"),
        row.get("prompt"),
        row.get("user_request"),
        row.get("input"),
        row.get("conversations"),
        row.get("messages"),
    )

    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    # Conversation-style rows.
    if isinstance(direct, list):
        for msg in direct:
            if isinstance(msg, dict):
                role = str(msg.get("from") or msg.get("role") or "").lower()
                content = msg.get("value") or msg.get("content")
                if content and role in {"human", "user"}:
                    return str(content).strip()
        if direct:
            return _json_preview(direct, max_chars=1200)

    deep = _deep_find(
        row,
        {"instruction", "query", "question", "prompt", "user_request", "input", "content", "value"},
    )
    if deep not in (None, "", [], {}):
        return str(deep).strip()

    return "Select and execute the appropriate tool call for this ToolBench task."


def _extract_answer(row: dict[str, Any]) -> tuple[str, ...]:
    candidates = [
        row.get("answer"),
        row.get("output"),
        row.get("response"),
        row.get("target"),
        row.get("solution"),
        row.get("final_answer"),
        row.get("api_path"),
        row.get("tool_calls"),
        row.get("tools"),
    ]

    deep = _deep_find(
        row,
        {"answer", "output", "response", "target", "solution", "final_answer", "api_path", "tool_calls"},
    )
    candidates.append(deep)

    out: list[str] = []
    for item in candidates:
        if item in (None, "", [], {}):
            continue
        if isinstance(item, str):
            out.append(item)
        else:
            out.append(_json_preview(item, max_chars=1000))

    deduped = []
    seen = set()
    for x in out:
        x = x.strip()
        if x and x.lower() != "none" and x not in seen:
            deduped.append(x)
            seen.add(x)

    return tuple(deduped) if deduped else ("tool_use_target_unavailable",)


def _extract_tool_evidence(row: dict[str, Any]) -> str:
    pieces = []

    for key in [
        "tools",
        "tool",
        "api",
        "apis",
        "api_list",
        "tool_list",
        "functions",
        "function",
        "category",
        "instruction",
        "query",
        "input",
        "conversations",
        "messages",
    ]:
        value = row.get(key)
        if value not in (None, "", [], {}):
            pieces.append(f"{key}: {_json_preview(value, max_chars=1800)}")

    if not pieces:
        pieces.append("raw_row: " + _json_preview(row, max_chars=2500))

    return "\n".join(pieces)


def _row_to_example(row: dict[str, Any], idx: int) -> QAExample:
    qid = str(
        _first_nonempty(
            row.get("id"),
            row.get("qid"),
            row.get("query_id"),
            row.get("toolbench_id"),
            f"toolbench_{idx}",
        )
    )

    instruction = _extract_instruction(row)
    answers = _extract_answer(row)
    evidence_text = _extract_tool_evidence(row)

    evidence = (
        EvidenceItem(
            id=f"{qid}_tool_context",
            title="ToolBench tool/API context",
            text=evidence_text,
            source_url=None,
            publisher="ToolBench",
            domain="toolbench",
            is_gold=True,
        ),
    )

    return QAExample(
        id=qid,
        question=instruction,
        gold_answers=answers,
        evidence=evidence,
        task_type="tool_use",
        meta={
            "dataset": "toolbench",
            "source": _PRIMARY_DATASET,
            "category": row.get("category"),
            "tool": row.get("tool") or row.get("tools"),
        },
    )


def _iter_toolbench_alternate(n: int | None = None, seed: int = 0) -> Iterator[QAExample]:
    templates = [
        (
            "Use a weather tool to answer: what is the current condition in Paris?",
            ("tool_call:weather", "Paris weather"),
            '{"tool": "weather", "required_args": ["location"], "returns": ["condition", "temperature"]}',
        ),
        (
            "Use a calculator tool to compute 17 multiplied by 23.",
            ("391", "tool_call:calculator"),
            '{"tool": "calculator", "required_args": ["expression"], "returns": ["value"]}',
        ),
        (
            "Use a search tool to find the capital of Japan.",
            ("Tokyo", "tool_call:search"),
            '{"tool": "search", "required_args": ["query"], "returns": ["snippet", "source"]}',
        ),
    ]

    total = 500 if n is None else n
    for i in range(total):
        q, answers, schema = templates[i % len(templates)]
        yield QAExample(
            id=f"toolbench_alternate_{seed}_{i}",
            question=q,
            gold_answers=answers,
            evidence=(
                EvidenceItem(
                    id=f"toolbench_alternate_{seed}_{i}_schema",
                    title="Tool schema",
                    text=schema,
                    source_url=None,
                    publisher="toolbench_alternate",
                    domain="tools.local",
                    is_gold=True,
                ),
            ),
            task_type="tool_use",
            meta={"dataset": "toolbench", "alternate": True},
        )


def _load_primary(split: str, streaming: bool):
    from datasets import load_dataset

    attempts = [
        (_PRIMARY_DATASET, split),
        (_PRIMARY_DATASET, "train"),
    ]

    last_exc: Exception | None = None
    for name, split_name in attempts:
        try:
            return load_dataset(name, split=split_name, streaming=streaming)
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unable to load ToolBench source.")


def iter_toolbench(
    *,
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
) -> Iterator[QAExample]:
    try:
        ds = _load_primary(split=split, streaming=streaming)
    except Exception:
        if _allow_dataset_alternate():
            yield from _iter_toolbench_alternate(n=n, seed=seed)
            return
        raise

    if streaming:
        try:
            ds = ds.shuffle(seed=seed, buffer_size=1024)
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
