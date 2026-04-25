"""
Dataset loaders for PCG-MAS experiments.

All loaders return a stream (`Iterator[QAExample]`) — they NEVER write the full
dataset to local disk. We rely on `datasets.load_dataset(..., streaming=True)`
for HF datasets and on small in-memory synthetic data for smoke tests.

Available datasets:
    - synthetic    : 50 in-memory multi-hop QA examples (no download)
    - hotpotqa     : HotpotQA distractor split (HF: hotpot_qa)
    - twowiki      : 2WikiMultihopQA (HF: 2wikimultihopqa)
    - toolbench    : ToolBench G1 subset (HF: ToolBench)

Each loader returns objects of type `QAExample` (defined in `base.py`), which
the agent layer consumes uniformly. This decoupling means the Prover, Verifier
and experiment scripts never need to know which dataset they're operating on.
"""
from __future__ import annotations

from pcg.datasets.base import EvidenceItem, QAExample, load_dataset_by_name

__all__ = [
    "EvidenceItem",
    "QAExample",
    "load_dataset_by_name",
]
