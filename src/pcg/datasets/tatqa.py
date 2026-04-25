"""
TAT-QA streaming loader.

TAT-QA (Tabular And Textual question answering) tests numerical and
structured grounding on financial reports. Each "passage" is a
(table, paragraphs) pair — typically a financial statement plus its
explanatory disclosures — and carries multiple questions of varying
type (span / arithmetic / count). We flatten so each yielded
QAExample is a single question-answer pair, with the table serialized
as a markdown-like string and each paragraph as a separate evidence
item. Per-question gold paragraphs (`rel_paragraphs`) are propagated
into `EvidenceItem.is_gold`.

The HF dataset uses a custom loading script, so we pass
`trust_remote_code=True`. The data is a static release; the script
just unpacks the JSON shipped with the repo.

References:
    - HF page: https://huggingface.co/datasets/next-tat/TAT-QA
    - Paper: Zhu et al., 2021, "TAT-QA: A Question Answering Benchmark
      on a Hybrid of Tabular and Textual Content in Finance"
"""
from __future__ import annotations

from typing import Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "next-tat/TAT-QA"
_DEFAULT_REVISION = "main"


def _serialize_table(table: object) -> str:
    """Render a TAT-QA table dict to markdown-like rows."""
    if isinstance(table, dict):
        rows = table.get("table") or []
    elif isinstance(table, list):
        rows = table
    else:
        return ""
    if not rows:
        return ""
    return "\n".join(" | ".join(str(c) for c in r) for r in rows)


def _normalize_answer(ans: object) -> tuple[str, ...]:
    if isinstance(ans, list):
        if not ans:
            return ("",)
        return tuple(str(a) for a in ans)
    return (str(ans) if ans is not None else "",)


def _row_to_examples(row: dict, base_idx: int) -> Iterator[QAExample]:
    """Each TAT-QA row is one (table, paragraphs) context with N questions.

    Flatten into N QAExamples sharing the evidence pool but each carrying
    its own per-question gold annotation."""
    table = row.get("table", {}) or {}
    paragraphs = row.get("paragraphs", []) or []
    table_text = _serialize_table(table)

    table_uid = (
        table.get("uid") if isinstance(table, dict) else None
    ) or f"tatqa_t{base_idx}"

    # Build the candidate evidence pool once.
    base_evidence: list[EvidenceItem] = []
    if table_text:
        base_evidence.append(EvidenceItem(
            id=f"{table_uid}_table",
            title=f"Table {table_uid}",
            text=table_text,
            source_url=None,
            publisher="tatqa",
            domain="tatqa",
            is_gold=False,  # gold flag is per-question, set below
        ))
    para_uids: list[str] = []
    for k, p in enumerate(paragraphs):
        if isinstance(p, dict):
            p_text = p.get("text", "")
            p_uid = p.get("uid") or f"{table_uid}_p{k}"
        else:
            p_text = str(p)
            p_uid = f"{table_uid}_p{k}"
        para_uids.append(str(p_uid))
        base_evidence.append(EvidenceItem(
            id=str(p_uid),
            title=f"Paragraph {k}",
            text=str(p_text),
            source_url=None,
            publisher="tatqa",
            domain="tatqa",
            is_gold=False,
        ))

    questions = row.get("questions", []) or []
    for q_i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        rel_paragraphs = set(str(x) for x in (q.get("rel_paragraphs") or []))
        # Mark evidence items as gold based on per-question rel_paragraphs.
        # The table is included as gold if the question's answer_from
        # field references it.
        answer_from = (q.get("answer_from") or "").lower()
        evidence: list[EvidenceItem] = []
        for ev in base_evidence:
            is_gold = False
            if ev.id in rel_paragraphs:
                is_gold = True
            elif "table" in answer_from and ev.id.endswith("_table"):
                is_gold = True
            evidence.append(EvidenceItem(
                id=ev.id, title=ev.title, text=ev.text,
                source_url=ev.source_url, publisher=ev.publisher,
                domain=ev.domain, is_gold=is_gold,
            ))

        qid = q.get("uid") or f"tatqa_{base_idx}_q{q_i}"
        yield QAExample(
            id=str(qid),
            question=str(q.get("question", "")),
            gold_answers=_normalize_answer(q.get("answer", "")),
            evidence=tuple(evidence),
            task_type="table_qa",
            meta={
                "answer_type": q.get("answer_type"),
                "answer_from": q.get("answer_from"),
                "scale": q.get("scale"),
                "derivation": q.get("derivation"),
            },
        )


def iter_tatqa(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    revision: str = _DEFAULT_REVISION,
    shuffle_buffer: int = 128,
) -> Iterator[QAExample]:
    """Yield TAT-QA examples flattened to one (question, ctx) pair each."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "huggingface `datasets` not installed. Install via "
            "`pip install datasets`."
        ) from exc

    ds = load_dataset(
        _DATASET_NAME,
        split=split,
        streaming=streaming,
        revision=revision,
        trust_remote_code=True,
    )
    if streaming and shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    count = 0
    for idx, row in enumerate(ds):
        for ex in _row_to_examples(row, idx):
            if n is not None and count >= n:
                return
            yield ex
            count += 1
