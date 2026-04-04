from __future__ import annotations

from src.data.medqa_adapter import load_medqa_examples
from src.data.medmcqa_adapter import load_medmcqa_examples


def load_examples(dataset_name: str, max_examples: int):
    dataset_name = dataset_name.lower()

    if dataset_name == "medqa":
        return load_medqa_examples(max_examples=max_examples)

    if dataset_name == "medmcqa":
        return load_medmcqa_examples(max_examples=max_examples)

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")
