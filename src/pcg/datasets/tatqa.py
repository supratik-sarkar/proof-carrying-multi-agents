"""
TAT-QA loader.

TAT-QA is table-and-text QA. Hugging Face variants often store one
table/document row with a nested `questions` list. This loader expands each
question into a separate QAExample and attaches the parent table + paragraphs
as evidence.
"""
from __future__ import annotations

from typing import Any, Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "next-tat/TAT-QA"


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


def _stringify_answer(answer: Any) -> tuple[str, ...]:
    if answer is None:
        return tuple()

    if isinstance(answer, list):
        return tuple(str(x) for x in answer if x not in (None, ""))

    if isinstance(answer, dict):
        for key in ["answer", "answers", "value", "spans", "answer_text"]:
            if key in answer:
                return _stringify_answer(answer[key])
        return (str(answer),)

    text = str(answer)
    return (text,) if text else tuple()


def _render_table(table: Any) -> str:
    if table is None:
        return ""

    if isinstance(table, dict):
        # Common variants:
        # {"table": [[...], ...]}
        # {"table_array": [[...], ...]}
        # {"header": [...], "rows": [...]}
        inner = _first_nonempty(
            table.get("table"),
            table.get("table_array"),
            table.get("rows"),
            table.get("data"),
        )

        header = _first_nonempty(table.get("header"), table.get("headers"))
        pieces = []
        if header:
            pieces.append("header: " + " | ".join(str(x) for x in _as_list(header)))
        if inner is not None:
            rendered = _render_table(inner)
            if rendered:
                pieces.append(rendered)
        if pieces:
            return "\n".join(pieces)

        return "\n".join(f"{k}: {v}" for k, v in table.items())

    if isinstance(table, list):
        rows = []
        for i, row in enumerate(table):
            if isinstance(row, list):
                rows.append(f"row {i}: " + " | ".join(str(c).strip() for c in row))
            elif isinstance(row, dict):
                rows.append(f"row {i}: " + " | ".join(f"{k}={v}" for k, v in row.items()))
            else:
                rows.append(f"row {i}: {row}")
        return "\n".join(rows)

    return str(table)


def _render_paragraphs(paragraphs: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    if isinstance(paragraphs, dict):
        # Some variants store {uid: text} or {"text": ...}
        if "text" in paragraphs or "paragraph" in paragraphs:
            text = str(paragraphs.get("text") or paragraphs.get("paragraph") or "")
            if text.strip():
                out.append(("paragraph", text.strip()))
            return out

        for k, v in paragraphs.items():
            text = str(v)
            if text.strip():
                out.append((str(k), text.strip()))
        return out

    for i, para in enumerate(_as_list(paragraphs)):
        if isinstance(para, dict):
            title = str(
                para.get("order")
                or para.get("uid")
                or para.get("title")
                or f"paragraph_{i}"
            )
            text = str(
                para.get("text")
                or para.get("paragraph")
                or para.get("content")
                or ""
            )
        else:
            title = f"paragraph_{i}"
            text = str(para)

        if text.strip():
            out.append((title, text.strip()))

    return out


def _make_evidence(parent_row: dict[str, Any], qid: str) -> tuple[EvidenceItem, ...]:
    table = _first_nonempty(
        parent_row.get("table"),
        parent_row.get("table_array"),
        parent_row.get("table_text"),
        parent_row.get("table_content"),
    )

    paragraphs = _first_nonempty(
        parent_row.get("paragraphs"),
        parent_row.get("paragraph"),
        parent_row.get("text"),
        parent_row.get("passages"),
    )

    evidence: list[EvidenceItem] = []

    table_text = _render_table(table)
    if table_text.strip():
        evidence.append(
            EvidenceItem(
                id=f"{qid}_table",
                title="TAT-QA table",
                text=table_text,
                source_url=None,
                publisher="TAT-QA",
                domain="tatqa",
                is_gold=True,
            )
        )

    for k, (title, text) in enumerate(_render_paragraphs(paragraphs)):
        evidence.append(
            EvidenceItem(
                id=f"{qid}_para{k}",
                title=f"TAT-QA {title}",
                text=text,
                source_url=None,
                publisher="TAT-QA",
                domain="tatqa",
                is_gold=True,
            )
        )

    if not evidence:
        evidence.append(
            EvidenceItem(
                id=f"{qid}_row",
                title="TAT-QA raw row",
                text=str(parent_row),
                source_url=None,
                publisher="TAT-QA",
                domain="tatqa",
                is_gold=False,
            )
        )

    return tuple(evidence)


def _question_to_example(
    parent_row: dict[str, Any],
    question_row: dict[str, Any],
    parent_idx: int,
    question_idx: int,
) -> QAExample:
    qid = str(
        question_row.get("uid")
        or question_row.get("id")
        or question_row.get("question_id")
        or f"tatqa_{parent_idx}_{question_idx}"
    )

    question = str(
        question_row.get("question")
        or question_row.get("query")
        or question_row.get("question_text")
        or ""
    ).strip()

    answer = _first_nonempty(
        question_row.get("answer"),
        question_row.get("answers"),
        question_row.get("gold_answer"),
        question_row.get("answer_text"),
    )

    meta = {
        "dataset": "tatqa",
        "answer_type": question_row.get("answer_type"),
        "answer_from": question_row.get("answer_from"),
        "scale": question_row.get("scale"),
        "derivation": question_row.get("derivation"),
        "table_uid": parent_row.get("table_uid") or parent_row.get("uid"),
        "uid": question_row.get("uid"),
        "parent_uid": parent_row.get("uid") or parent_row.get("table_uid"),
    }

    return QAExample(
        id=qid,
        question=question,
        gold_answers=_stringify_answer(answer),
        evidence=_make_evidence(parent_row, qid),
        task_type="table_qa",
        meta=meta,
    )


def _row_to_examples(row: dict[str, Any], parent_idx: int) -> list[QAExample]:
    """Expand a TAT-QA parent row into one or more QAExamples."""
    questions = _first_nonempty(
        row.get("questions"),
        row.get("qa_pairs"),
        row.get("qas"),
        row.get("question_answer_pairs"),
    )

    # Correct TAT-QA path: one parent row has many nested question rows.
    if isinstance(questions, list) and questions:
        out = []
        for q_idx, q in enumerate(questions):
            if isinstance(q, dict):
                out.append(_question_to_example(row, q, parent_idx, q_idx))
        return out

    # Fallback for flat variants where each row is already one QA pair.
    flat_question = str(
        row.get("question")
        or row.get("query")
        or row.get("question_text")
        or ""
    ).strip()

    flat_answer = _first_nonempty(
        row.get("answer"),
        row.get("answers"),
        row.get("gold_answer"),
        row.get("answer_text"),
    )

    qid = str(
        row.get("uid")
        or row.get("id")
        or row.get("question_id")
        or row.get("table_uid")
        or f"tatqa_{parent_idx}"
    )

    return [
        QAExample(
            id=qid,
            question=flat_question,
            gold_answers=_stringify_answer(flat_answer),
            evidence=_make_evidence(row, qid),
            task_type="table_qa",
            meta={
                "dataset": "tatqa",
                "answer_type": row.get("answer_type"),
                "answer_from": row.get("answer_from"),
                "scale": row.get("scale"),
                "derivation": row.get("derivation"),
                "table_uid": row.get("table_uid"),
                "uid": row.get("uid"),
            },
        )
    ]


def iter_tatqa(
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
    for parent_idx, row in enumerate(ds):
        for ex in _row_to_examples(row, parent_idx):
            if n is not None and count >= n:
                return
            yield ex
            count += 1
