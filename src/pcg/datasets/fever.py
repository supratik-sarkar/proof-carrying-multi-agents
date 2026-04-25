"""
FEVER streaming loader.

FEVER (Fact Extraction and VERification) examples are claims labeled
SUPPORTS / REFUTES / NOT ENOUGH INFO with evidence sentence-IDs from
Wikipedia. This is the most directly aligned external benchmark for
PCG-MAS: every accepted "claim" must be backed by retrievable evidence
or labeled NEI — exactly the contract the certificate Z encodes.

Note: the canonical FEVER HF dataset only ships evidence references
(URL + sentence-id); the underlying Wikipedia text must be retrieved
separately by the Prover's retriever. For local debugging / smoke tests
we synthesize a placeholder evidence text from the URL so the pipeline
runs end-to-end. For paper-grade evaluation, wire in a real Wikipedia
retrieval step (we suggest `wiki_dpr` or a local FEVER Wikipedia dump).

References:
    - HF page: https://huggingface.co/datasets/fever
    - Paper: Thorne et al., 2018, "FEVER: a Large-scale Dataset for Fact
      Extraction and VERification"
"""
from __future__ import annotations

from typing import Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "fever"
_DATASET_CONFIG = "v1.0"
_DEFAULT_REVISION = "main"


def _flatten_evidence(raw: object) -> list[tuple[str, int]]:
    """FEVER evidence is a list-of-list-of-[ann_id, ev_id, wiki_url, sent_id].

    Returns a flat list of (wiki_url, sent_id). Skips NEI / empty entries.
    """
    out: list[tuple[str, int]] = []
    if not isinstance(raw, list):
        return out
    for ev_set in raw:
        if not isinstance(ev_set, list):
            continue
        for ev in ev_set:
            if isinstance(ev, list) and len(ev) >= 4:
                wiki_url = ev[2]
                sent_id = ev[3]
                if wiki_url is None:
                    continue
                try:
                    out.append((str(wiki_url), int(sent_id)))
                except (TypeError, ValueError):
                    continue
            elif isinstance(ev, dict):
                wiki_url = ev.get("wikipedia_url") or ev.get("page")
                sent_id = ev.get("sentence_id")
                if wiki_url is None or sent_id is None:
                    continue
                try:
                    out.append((str(wiki_url), int(sent_id)))
                except (TypeError, ValueError):
                    continue
    return out


def _row_to_example(row: dict, idx: int) -> QAExample:
    """Convert a FEVER row to a QAExample.

    FEVER schema (labelled_dev split):
        {
          "id": int,
          "label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
          "claim": str,
          "evidence_annotation_id": int,
          "evidence_id": int,
          "evidence_wiki_url": str,
          "evidence_sentence_id": int,
        }
    """
    qid = f"fever_{row.get('id', idx)}"
    claim = row.get("claim", "")
    label = row.get("label", "NOT ENOUGH INFO")

    # Newer dumps flatten evidence to top-level scalars; older ones nest.
    evidence_pairs: list[tuple[str, int]] = []
    if "evidence" in row:
        evidence_pairs = _flatten_evidence(row.get("evidence"))
    else:
        wu = row.get("evidence_wiki_url")
        sid = row.get("evidence_sentence_id")
        if wu is not None and sid is not None:
            try:
                evidence_pairs = [(str(wu), int(sid))]
            except (TypeError, ValueError):
                evidence_pairs = []

    # Build evidence items. For NEI claims with no evidence, attach a
    # single sentinel "no-evidence" item so the pipeline still has a
    # candidate set; CovGap will catch it.
    evidence: list[EvidenceItem] = []
    if not evidence_pairs:
        evidence.append(EvidenceItem(
            id=f"{qid}_noev",
            title="(no evidence retrieved)",
            text="",
            source_url=None,
            publisher="wikipedia",
            domain="en.wikipedia.org",
            is_gold=False,
        ))
    else:
        for k, (wu, sid) in enumerate(evidence_pairs):
            url = (
                wu if str(wu).startswith("http")
                else f"https://en.wikipedia.org/wiki/{wu}"
            )
            placeholder = (
                f"[FEVER evidence pointer — retrieve from {url} sentence {sid}. "
                "For paper-grade eval, wire in real Wikipedia retrieval here.]"
            )
            evidence.append(EvidenceItem(
                id=f"{qid}_e{k}",
                title=str(wu).replace("_", " "),
                text=placeholder,
                source_url=url,
                publisher="wikipedia",
                domain="en.wikipedia.org",
                is_gold=True,  # FEVER's annotated evidence is by definition gold
            ))

    return QAExample(
        id=qid,
        question=str(claim),
        gold_answers=(str(label),),
        evidence=tuple(evidence),
        task_type="qa",
        meta={
            "label": label,
            "verifiable": row.get("verifiable", "UNVERIFIABLE"),
            "fever_id": row.get("id"),
        },
    )


def iter_fever(
    *,
    split: str = "labelled_dev",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    revision: str = _DEFAULT_REVISION,
    shuffle_buffer: int = 1024,
) -> Iterator[QAExample]:
    """Yield FEVER examples (v1.0 config).

    Args:
        split: HF split. FEVER's eval split is `labelled_dev`.
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
        _DATASET_CONFIG,
        split=split,
        streaming=streaming,
        revision=revision,
        trust_remote_code=True,
    )
    if streaming and shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    count = 0
    for idx, row in enumerate(ds):
        if n is not None and count >= n:
            break
        yield _row_to_example(row, idx)
        count += 1
