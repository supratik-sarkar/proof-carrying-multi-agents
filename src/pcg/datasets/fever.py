"""
FEVER claim-verification loader.

Source: copenlu/fever_gold_evidence (HF Hub).

Why this source: the canonical fever/fever HF dataset is script-based and is
no longer loadable under modern datasets (>=3.0). The previous BeIR/fever
fallback is a retrieval dataset with no FEVER labels — every gold answer
defaulted to "SUPPORTS", making the f1-vs-gold metric meaningless.
copenlu/fever_gold_evidence carries the real three-class labels
(SUPPORTS / REFUTES / NOT ENOUGH INFO) and gold evidence with Wikipedia
page ids, which is exactly what claim verification needs.

Row schema (validation split):
    claim       : str         the claim under verification
    label       : str         SUPPORTS | REFUTES | NOT ENOUGH INFO
    evidence    : list        [[wiki_page_id, sent_idx, sentence_text, _], ...]
    id          : str
    verifiable  : str         VERIFIABLE | NOT VERIFIABLE
    original_id : int         original FEVER claim id
"""
from __future__ import annotations

import os
from typing import Any, Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_HF_DATASET = "copenlu/fever_gold_evidence"


def _allow_dataset_alternate() -> bool:
    return os.environ.get("PCG_ALLOW_DATASET_ALTERNATE", "0") == "1"


def _iter_fever_alternate(n: int | None = None, seed: int = 0) -> Iterator[QAExample]:
    """Deterministic FEVER-shaped alternate for environment preflight only."""
    base = [
        ("The Eiffel Tower is located in Paris.", "SUPPORTS",
         "The Eiffel Tower is a wrought-iron tower on the Champ de Mars in Paris, France.",
         "Eiffel_Tower"),
        ("The Pacific Ocean is smaller than the Arctic Ocean.", "REFUTES",
         "The Pacific Ocean is the largest and deepest of Earths oceanic divisions.",
         "Pacific_Ocean"),
        ("Marie Curie won a Nobel Prize.", "SUPPORTS",
         "Marie Curie was awarded Nobel Prizes in Physics and Chemistry.",
         "Marie_Curie"),
        ("Mount Everest is in Antarctica.", "REFUTES",
         "Mount Everest is Earths highest mountain above sea level, located in the Himalayas.",
         "Mount_Everest"),
        ("Jane Austen wrote Pride and Prejudice.", "SUPPORTS",
         "Pride and Prejudice is an 1813 novel of manners by Jane Austen.",
         "Pride_and_Prejudice"),
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
                    title=title, text=text,
                    source_url=None, publisher="alternate_fever",
                    domain="fever.local", is_gold=True,
                ),
            ),
            task_type="fact_verification",
            meta={"dataset": "fever", "alternate": True, "label": label},
        )


def _parse_evidence(raw: Any) -> list[tuple[str, str]]:
    """Return list of (title, sentence_text) tuples from copenlu evidence field.

    The evidence field is a list of [wiki_page_id, sent_idx, sentence_text, _]
    quadruples. We deduplicate by (page_id, sent_idx) and return display-ready
    (title, text) pairs.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, int]] = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        page = str(item[0]) if item[0] is not None else ""
        try:
            sidx = int(item[1]) if item[1] is not None else -1
        except Exception:
            sidx = -1
        sent = str(item[2]) if item[2] is not None else ""
        key = (page, sidx)
        if key in seen or not sent.strip():
            continue
        seen.add(key)
        title = page.replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")")
        out.append((title, sent))
    return out


def _row_to_example(row: dict[str, Any], idx: int) -> QAExample:
    qid = str(row.get("id") or row.get("original_id") or f"fever_{idx}")
    claim = str(row.get("claim") or "").strip()
    label = str(row.get("label") or "NOT ENOUGH INFO").strip().upper()
    if label not in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}:
        label = "NOT ENOUGH INFO"

    ev_pairs = _parse_evidence(row.get("evidence"))
    evidence_items: list[EvidenceItem] = []
    for i, (title, text) in enumerate(ev_pairs[:8]):    # cap evidence pool size
        evidence_items.append(EvidenceItem(
            id=f"{qid}_e{i}",
            title=title or "FEVER evidence",
            text=text,
            source_url=None,
            publisher="copenlu/fever_gold_evidence",
            domain="wikipedia",
            is_gold=True,
        ))
    if not evidence_items:
        # NEI claims may have empty evidence — keep a placeholder so retrieval
        # pool is never empty, but mark non-gold.
        evidence_items.append(EvidenceItem(
            id=f"{qid}_e0",
            title="No evidence",
            text=claim,
            source_url=None,
            publisher="copenlu/fever_gold_evidence",
            domain="wikipedia",
            is_gold=False,
        ))

    return QAExample(
        id=qid,
        question=claim,
        gold_answers=(label,),
        evidence=tuple(evidence_items),
        task_type="fact_verification",
        meta={"dataset": "fever", "source": _HF_DATASET,
              "verifiable": row.get("verifiable")},
    )


def iter_fever(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    shuffle_buffer: int = 1024,
) -> Iterator[QAExample]:
    """Yield FEVER claim-verification examples from copenlu/fever_gold_evidence."""
    from datasets import load_dataset

    try:
        ds = load_dataset(_HF_DATASET, split=split, streaming=streaming)
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
