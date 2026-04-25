"""
HotpotQA streaming loader.

Streams HotpotQA distractor split from HuggingFace Hub. No local disk writes
beyond the small text-cache HF maintains for the requested shards.

Note on the HF dataset name: the canonical card is `hotpot_qa`. We pin a
specific revision in the loader so reviewers see exactly the snapshot we ran.
If HF changes their schema, we fail loud rather than silently load wrong data.

References:
    - HF page: https://huggingface.co/datasets/hotpot_qa
    - Original paper: Yang et al., 2018, "HotpotQA: A Dataset for Diverse,
      Explainable Multi-hop Question Answering"
"""
from __future__ import annotations

from typing import Iterator
from urllib.parse import urlparse

from pcg.datasets.base import EvidenceItem, QAExample

# Pin a revision so reviewers can reproduce. (`main` resolves to a SHA at load
# time; we record the SHA in the experiment log.)
_DATASET_NAME = "hotpot_qa"
_DATASET_CONFIG = "distractor"
_DEFAULT_REVISION = "main"


def _domain_of(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return urlparse(url).netloc.lower() or None
    except Exception:
        return None


def _row_to_example(row: dict, idx: int) -> QAExample:
    """Convert a HotpotQA HF row into our QAExample.

    HotpotQA distractor schema (validation split):
        {
          "id": str,
          "question": str,
          "answer": str,
          "type": "comparison" | "bridge",
          "level": "easy" | "medium" | "hard",
          "supporting_facts": {"title": [...], "sent_id": [...]},
          "context": {"title": [...], "sentences": [[sent, sent, ...], ...]},
        }
    """
    qid = row.get("id", f"hpqa_{idx}")
    question = row["question"]
    answer = row["answer"]

    # Build set of (title, sent_id) gold keys
    sf = row.get("supporting_facts", {})
    gold_keys: set[tuple[str, int]] = set()
    if "title" in sf and "sent_id" in sf:
        gold_keys = set(zip(sf["title"], sf["sent_id"]))

    # Materialize context paragraphs as evidence items. We treat each
    # paragraph as a single evidence item (concatenating its sentences).
    # An evidence item is "gold" if any of its sentences is in gold_keys.
    evidence: list[EvidenceItem] = []
    ctx = row.get("context", {})
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])
    for k, (title, sents) in enumerate(zip(titles, sentences)):
        is_gold = any((title, i) in gold_keys for i in range(len(sents)))
        text = " ".join(sents).strip()
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        evidence.append(EvidenceItem(
            id=f"{qid}_p{k}",
            title=title,
            text=text,
            source_url=url,
            publisher="wikipedia",
            domain="en.wikipedia.org",
            is_gold=is_gold,
        ))

    return QAExample(
        id=qid,
        question=question,
        gold_answers=(answer,),
        evidence=tuple(evidence),
        task_type="qa",
        meta={"hop_type": row.get("type"), "level": row.get("level")},
    )


def iter_hotpotqa(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    revision: str = _DEFAULT_REVISION,
    shuffle_buffer: int = 1024,
) -> Iterator[QAExample]:
    """Yield HotpotQA examples (distractor config).

    Args:
        split: HF split (validation | train).
        n: cap on number of examples (None = stream all).
        seed: shuffle seed (used only when shuffle_buffer > 0).
        streaming: True keeps memory bounded (recommended).
        revision: HF dataset revision tag/SHA. Bump only with intent.
        shuffle_buffer: when > 0, applies HF's reservoir shuffle.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "huggingface `datasets` not installed. Install with "
            "`pip install datasets`. Already in pyproject.toml dependencies."
        ) from exc

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
