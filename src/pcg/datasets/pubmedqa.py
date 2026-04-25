"""
PubMedQA streaming loader.

PubMedQA tests biomedical research question answering. Each example pairs
a research question with a paper abstract context and a final yes/no/maybe
decision plus a long-form rationale. Tests PCG-MAS on high-stakes
scientific grounding where every accepted claim should be auditable
against the abstract — exactly the failure mode that hurts most in
clinical / safety-critical deployment.

We use the `pqa_labeled` config (1k expert-annotated examples) by default;
swap to `pqa_artificial` (~211k auto-generated) for larger sweeps. The
`pqa_unlabeled` config has no final_decision and is unsuitable for
benchmarking under our protocol.

References:
    - HF page: https://huggingface.co/datasets/qiaojin/PubMedQA
    - Paper: Jin et al., 2019, "PubMedQA: A Dataset for Biomedical
      Research Question Answering"
"""
from __future__ import annotations

from typing import Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "qiaojin/PubMedQA"
_DATASET_CONFIG = "pqa_labeled"
_DEFAULT_REVISION = "main"


def _row_to_example(row: dict, idx: int) -> QAExample:
    """Convert a PubMedQA row to a QAExample.

    PubMedQA `pqa_labeled` schema:
        {
          "pubid": int,
          "question": str,
          "context": {
            "contexts": [str, str, ...],   # paragraphs of the abstract
            "labels":   [str, str, ...],   # section labels (BACKGROUND, etc.)
            "meshes":   [str, ...],        # MeSH headings
          },
          "long_answer": str,
          "final_decision": "yes" | "no" | "maybe",
          "year": str,
        }
    """
    qid = f"pubmedqa_{row.get('pubid', idx)}"
    question = row.get("question", "")
    decision = row.get("final_decision", "")

    ctx = row.get("context", {}) or {}
    paragraphs = []
    section_labels = []
    if isinstance(ctx, dict):
        paragraphs = list(ctx.get("contexts") or [])
        section_labels = list(ctx.get("labels") or [])

    evidence: list[EvidenceItem] = []
    if not paragraphs:
        # Defensive fallback: use the entire context as one item
        evidence.append(EvidenceItem(
            id=f"{qid}_abs",
            title=str(row.get("pubid", idx)),
            text=str(ctx),
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{row.get('pubid', '')}/",
            publisher="pubmed",
            domain="pubmed.ncbi.nlm.nih.gov",
            is_gold=True,
        ))
    else:
        for k, p_text in enumerate(paragraphs):
            section = (
                section_labels[k] if k < len(section_labels) else f"section_{k}"
            )
            evidence.append(EvidenceItem(
                id=f"{qid}_p{k}",
                title=f"PMID {row.get('pubid', idx)} · {section}",
                text=str(p_text),
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{row.get('pubid', '')}/",
                publisher="pubmed",
                domain="pubmed.ncbi.nlm.nih.gov",
                # All paragraphs of the cited abstract count as gold for
                # the question — PubMedQA does not annotate per-sentence
                # support, so this is the right granularity.
                is_gold=True,
            ))

    return QAExample(
        id=qid,
        question=str(question),
        gold_answers=(str(decision),),
        evidence=tuple(evidence),
        task_type="qa",
        meta={
            "long_answer": row.get("long_answer", ""),
            "year": row.get("year", ""),
            "meshes": (ctx.get("meshes") or []) if isinstance(ctx, dict) else [],
        },
    )


def iter_pubmedqa(
    *,
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    revision: str = _DEFAULT_REVISION,
    config: str = _DATASET_CONFIG,
    shuffle_buffer: int = 256,
) -> Iterator[QAExample]:
    """Yield PubMedQA examples.

    Args:
        config: "pqa_labeled" (1k, default) | "pqa_artificial" (~211k).
        split: HF split. Most PubMedQA configs only have `train`.
        n, seed, streaming, revision, shuffle_buffer: same as hotpotqa.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "huggingface `datasets` not installed. Install via "
            "`pip install datasets`."
        ) from exc

    ds = load_dataset(
        _DATASET_NAME,
        config,
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
