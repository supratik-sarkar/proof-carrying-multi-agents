"""
ToolBench (G1 subset) streaming loader.

ToolBench is the canonical agentic tool-use dataset. We use the G1 (single-tool)
subset because it's the cleanest fit for our certificate machinery — each
question has exactly one tool call whose output becomes the supporting
evidence. G2/G3 (multi-tool, multi-step) are included as appendix experiments
under the same loader API.

This loader is intentionally tolerant: the HF mirror schemas drift, and we
prefer skipping malformed rows over failing the run.

References:
    - Original repo: https://github.com/OpenBMB/ToolBench
    - HF mirror: https://huggingface.co/datasets/ToolBench/ToolBench
"""
from __future__ import annotations

from typing import Iterator

from pcg.datasets.base import EvidenceItem, QAExample

_DATASET_NAME = "ToolBench/ToolBench"
_DEFAULT_REVISION = "main"


def _row_to_example(row: dict, idx: int) -> QAExample:
    qid = row.get("query_id") or row.get("id") or f"toolbench_{idx}"
    question = row.get("query") or row.get("question") or ""

    # ToolBench answers are dispatch traces; we use the final assistant turn
    # as the canonical answer string when available.
    answer = ""
    if "answer" in row:
        answer = str(row["answer"])
    elif "answer_generation" in row:
        ag = row["answer_generation"]
        if isinstance(ag, dict):
            answer = ag.get("final_answer", "") or ""

    # Evidence = the API responses returned by tool calls during the trace.
    evidence: list[EvidenceItem] = []
    if "answer_generation" in row and isinstance(row["answer_generation"], dict):
        msgs = row["answer_generation"].get("train_messages", []) or []
        for k, msg in enumerate(msgs):
            if msg.get("role") == "function":
                tool_name = msg.get("name", "tool")
                content = str(msg.get("content", ""))
                evidence.append(EvidenceItem(
                    id=f"{qid}_t{k}",
                    title=f"tool_response::{tool_name}",
                    text=content[:4000],   # cap to keep memory bounded
                    source_url=None,
                    publisher="toolbench_api",
                    domain="rapidapi.com",
                    is_gold=True,    # tool outputs are by construction the support
                ))

    tools_available = []
    if "available_tools" in row and isinstance(row["available_tools"], list):
        tools_available = [t.get("name", "") for t in row["available_tools"] if isinstance(t, dict)]

    return QAExample(
        id=str(qid),
        question=question,
        gold_answers=(answer,) if answer else (),
        evidence=tuple(evidence),
        task_type="tool_use",
        meta={"tools_available": tools_available},
    )


def iter_toolbench(
    *,
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
    streaming: bool = True,
    revision: str = _DEFAULT_REVISION,
    shuffle_buffer: int = 1024,
    config: str = "G1_instruction",
) -> Iterator[QAExample]:
    """Yield ToolBench examples.

    Args:
        config: G1_instruction (single-tool, default), G2_instruction,
                G3_instruction. Reviewers asking about appendix tool-use
                experiments should consult the same loader with config=G2/G3.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("huggingface `datasets` not installed.") from exc

    try:
        ds = load_dataset(
            _DATASET_NAME, config,
            split=split, streaming=streaming, revision=revision,
            trust_remote_code=False,
        )
    except Exception:
        # Fall back to default config if the named one is unavailable on
        # the current HF mirror.
        ds = load_dataset(
            _DATASET_NAME,
            split=split, streaming=streaming, revision=revision,
            trust_remote_code=False,
        )

    if streaming and shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    count = 0
    for idx, row in enumerate(ds):
        if n is not None and count >= n:
            break
        try:
            ex = _row_to_example(row, idx)
        except Exception:
            continue
        if not ex.evidence and not ex.gold_answers:
            # Useless row, skip.
            continue
        yield ex
        count += 1
