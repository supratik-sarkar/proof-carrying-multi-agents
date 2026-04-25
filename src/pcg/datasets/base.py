"""
Common types and dispatch for dataset loaders.

The `QAExample` dataclass is the single interface the experiment scripts speak;
adding a new dataset means writing one loader that yields these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal


@dataclass(frozen=True)
class EvidenceItem:
    """One candidate evidence document.

    Fields:
        id:         stable doc id within the dataset
        title:      paragraph / page title (used for shingle alignment)
        text:       raw evidence text (UTF-8)
        source_url: optional URL or external identifier (Wikipedia, ArXiv, etc)
        publisher:  e.g. "wikipedia", "arxiv" — drives provenance labelling
        domain:     eTLD+1 of source_url, derived once at load time
        is_gold:    True iff the dataset annotates this passage as supporting
                    the answer (used for oracle baselines and CovGap analysis)
    """

    id: str
    title: str
    text: str
    source_url: str | None = None
    publisher: str | None = None
    domain: str | None = None
    is_gold: bool = False


@dataclass(frozen=True)
class QAExample:
    """One example. Common shape across HotpotQA / 2Wiki / TAT-QA / ToolBench.

    The `task_type` field distinguishes pure QA from agentic / tool-using
    examples so the Prover can branch.

    Fields:
        id:           stable example id
        question:     the question / instruction text
        gold_answers: list of acceptable answer strings (HotpotQA convention)
        evidence:     candidate evidence pool (distractors + gold)
        task_type:    "qa" | "table_qa" | "tool_use" | "web_action"
        meta:         dataset-specific extras (HotpotQA hop type, ToolBench
                      tool list, etc.)
    """

    id: str
    question: str
    gold_answers: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...]
    task_type: Literal["qa", "table_qa", "tool_use", "web_action"] = "qa"
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def load_dataset_by_name(
    name: str,
    *,
    split: str = "validation",
    n_examples: int | None = None,
    seed: int = 0,
    streaming: bool = True,
) -> Iterator[QAExample]:
    """Resolve `name` to the right loader and yield up to `n_examples`.

    Args:
        name: one of {"synthetic", "hotpotqa", "twowiki", "toolbench"}.
        split: dataset split. Synthetic ignores this.
        n_examples: cap on number of examples. None = stream the whole split.
        seed: shuffle seed (when implemented; HF streaming shuffles via buffer).
        streaming: forwarded to `datasets.load_dataset`. Synthetic always
                   in-memory; the flag is just for API uniformity.
    """
    if name == "synthetic":
        from pcg.datasets.synthetic import iter_synthetic
        yield from iter_synthetic(n=n_examples, seed=seed)
    elif name == "hotpotqa":
        from pcg.datasets.hotpotqa import iter_hotpotqa
        yield from iter_hotpotqa(split=split, n=n_examples, seed=seed, streaming=streaming)
    elif name == "twowiki":
        from pcg.datasets.twowiki import iter_twowiki
        yield from iter_twowiki(split=split, n=n_examples, seed=seed, streaming=streaming)
    elif name == "toolbench":
        from pcg.datasets.toolbench import iter_toolbench
        yield from iter_toolbench(split=split, n=n_examples, seed=seed, streaming=streaming)
    else:
        raise ValueError(
            f"Unknown dataset {name!r}. Known: synthetic, hotpotqa, twowiki, toolbench"
        )
