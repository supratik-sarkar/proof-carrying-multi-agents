"""
WebLINX streaming loader.

WebLINX records real-world web navigation traces — a user instruction
plus a turn-by-turn dialogue against a live page DOM. Each turn pairs
the latest utterances and DOM snapshot with a gold action (click,
type, submit, etc.). Tests PCG-MAS on agentic, multi-turn grounding
where evidence is dynamic state rather than static text.

The full DOM snapshots are large, so we truncate to a configurable
character budget. For paper-grade evaluation, swap to a smarter DOM
reduction (e.g. WebLINX's own truncated_html column when present) or
to the `dmr_truncated` config which ships pre-pruned snapshots.

References:
    - HF page: https://huggingface.co/datasets/McGill-NLP/WebLINX
    - Paper: Lù et al., 2024, "WebLINX: Real-World Website Navigation
      with Multi-Turn Dialogue"
"""
from __future__ import annotations

from typing import Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "McGill-NLP/WebLINX"
_DATASET_CONFIG = "default"
_DEFAULT_REVISION = "main"

# Cap DOM snapshot text by characters; ~8kB is enough for the visible
# region in most cases. Bump if your retriever / context can take more.
_DOM_CHAR_BUDGET = 8000


def _utterances_text(utterances: object) -> str:
    """Concatenate user-side utterances into a single instruction string."""
    if not isinstance(utterances, list):
        return ""
    parts: list[str] = []
    for u in utterances:
        if isinstance(u, dict):
            speaker = (u.get("speaker") or u.get("role") or "").lower()
            text = u.get("text") or u.get("utterance") or ""
            if speaker in ("user", "navigator", "instructor"):
                parts.append(str(text))
        elif isinstance(u, str):
            parts.append(u)
    return " ".join(p for p in parts if p)


def _action_to_str(action: object) -> str:
    """Render a WebLINX action dict to a compact string the Verifier can match."""
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        intent = action.get("intent") or action.get("type") or "action"
        target = (
            action.get("element_text")
            or action.get("text")
            or action.get("element")
            or ""
        )
        return f"{intent}({target})".strip()
    return str(action)


def _row_to_example(row: dict, idx: int) -> QAExample:
    """Convert a WebLINX row to a QAExample.

    WebLINX schema (default config) — fields vary slightly by release;
    we read defensively and tolerate missing keys.
    """
    demo = row.get("demo_name") or row.get("demonstration") or "unknown"
    qid = f"weblinx_{demo}_{row.get('turn_index', idx)}"

    instruction = _utterances_text(row.get("utterances")) or row.get("instruction", "")
    action_str = _action_to_str(row.get("action"))

    # Evidence: the page DOM snapshot at decision time. We provide it as
    # one big evidence item; the retriever can chunk it before passing
    # to the Prover.
    dom = (
        row.get("snapshot")
        or row.get("html")
        or row.get("truncated_html")
        or ""
    )
    dom_text = str(dom)
    if len(dom_text) > _DOM_CHAR_BUDGET:
        dom_text = dom_text[:_DOM_CHAR_BUDGET] + "\n…[DOM truncated]"

    evidence: tuple[EvidenceItem, ...] = (
        EvidenceItem(
            id=f"{qid}_dom",
            title=f"DOM @ {demo} t{row.get('turn_index', idx)}",
            text=dom_text,
            source_url=row.get("url") or None,
            publisher="weblinx",
            domain="weblinx",
            is_gold=True,  # The DOM at decision time is the grounding signal
        ),
    )

    return QAExample(
        id=qid,
        question=str(instruction),
        gold_answers=(action_str,),
        evidence=evidence,
        task_type="web_action",
        meta={
            "demo_name": demo,
            "turn_index": row.get("turn_index"),
            "intent": row.get("intent"),
            "url": row.get("url"),
        },
    )


def iter_weblinx(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    revision: str = _DEFAULT_REVISION,
    config: str = _DATASET_CONFIG,
    shuffle_buffer: int = 256,
) -> Iterator[QAExample]:
    """Yield WebLINX examples (one per turn).

    Args:
        config: WebLINX HF config name. "default" works for most builds.
        split: HF split. WebLINX exposes train / validation / test_iid /
               test_visual / test_geo / test_cat / test_web — pick what you need.
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
