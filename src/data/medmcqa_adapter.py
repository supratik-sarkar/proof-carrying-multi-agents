from __future__ import annotations

from datasets import load_dataset


def load_medmcqa_examples(max_examples: int = 20):
    ds = load_dataset("medmcqa")

    if "validation" in ds:
        split = ds["validation"]
    elif "train" in ds:
        split = ds["train"]
    else:
        split = ds[list(ds.keys())[0]]

    examples = []
    for i, row in enumerate(split):
        if i >= max_examples:
            break

        choices = [
            row.get("opa", ""),
            row.get("opb", ""),
            row.get("opc", ""),
            row.get("opd", ""),
        ]

        examples.append(
            {
                "instance_id": str(row.get("id", i)),
                "question": row.get("question", ""),
                "choices": choices,
                "gold_answer": row.get("cop", None),
                "documents": [],
                "task_type": "mcq",
            }
        )

    return examples
