"""
2WikiMultihopQA streaming loader.

A more diverse multi-hop set than HotpotQA: includes inference, comparison,
compositional, and bridge-comparison hop types, with explicit reasoning
chains in the meta.

References:
    - HF page: https://huggingface.co/datasets/voidful/2WikiMultihopQA
      (community mirror; original is at the project page below)
    - Project page: https://github.com/Alab-NII/2wikimultihop
"""
from __future__ import annotations

from typing import Iterator

from pcg.datasets.base import EvidenceItem, QAExample

# We use the voidful mirror because it streams cleanly. Pin to a specific
# commit at experiment time (recorded in run logs).
_DATASET_NAME = "voidful/2WikiMultihopQA"
_DEFAULT_REVISION = "main"


def _row_to_example(row: dict, idx: int) -> QAExample:
    qid = row.get("_id") or row.get("id") or f"2wiki_{idx}"
    question = row["question"]
    answer = row["answer"]

    # 2Wiki context schema mirrors HotpotQA: {"title": [...], "content": [...]}
    # where each content is a list of sentences. Some mirrors flatten this, so
    # we handle both shapes.
    ctx = row.get("context", {})
    titles = ctx.get("title", [])
    contents = ctx.get("content") or ctx.get("sentences") or []
    sf = row.get("supporting_facts", {})
    gold_keys: set[str] = set(sf.get("title", []))

    evidence: list[EvidenceItem] = []
    for k, (title, sents_or_text) in enumerate(zip(titles, contents)):
        if isinstance(sents_or_text, list):
            text = " ".join(sents_or_text).strip()
        else:
            text = str(sents_or_text).strip()
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        evidence.append(EvidenceItem(
            id=f"{qid}_p{k}",
            title=title,
            text=text,
            source_url=url,
            publisher="wikipedia",
            domain="en.wikipedia.org",
            is_gold=(title in gold_keys),
        ))

    return QAExample(
        id=str(qid),
        question=question,
        gold_answers=(answer,),
        evidence=tuple(evidence),
        task_type="qa",
        meta={"hop_type": row.get("type"), "evidence_chain": row.get("evidences")},
    )


def iter_twowiki(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    revision: str = _DEFAULT_REVISION,
    shuffle_buffer: int = 1024,
) -> Iterator[QAExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("huggingface `datasets` not installed.") from exc

    ds = load_dataset(
        _DATASET_NAME,
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
        try:
            yield _row_to_example(row, idx)
            count += 1
        except (KeyError, TypeError):
            # Skip rows whose schema doesn't match — safer than crashing.
            continue
