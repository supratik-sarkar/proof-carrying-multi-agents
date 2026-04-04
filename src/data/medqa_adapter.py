from __future__ import annotations

from datasets import load_dataset


def load_medqa_examples(max_examples: int = 20):
    candidate_datasets = [
        "GBaker/MedQA-USMLE-4-options",
    ]

    ds = None
    last_error = None

    for name in candidate_datasets:
        try:
            ds = load_dataset(name)
            break
        except Exception as e:
            last_error = e

    if ds is None:
        raise RuntimeError(
            "Could not load MedQA dataset from Hugging Face. "
            f"Last error: {last_error}"
        )

    if "test" in ds:
        split = ds["test"]
    elif "validation" in ds:
        split = ds["validation"]
    else:
        split = ds["train"]

    examples = []
    for i, row in enumerate(split):
        if i >= max_examples:
            break

        options = row.get("options", [])
        if isinstance(options, dict):
            options = list(options.values())

        examples.append(
            {
                "instance_id": str(row.get("id", i)),
                "question": row.get("question", ""),
                "choices": options,
                "gold_answer": row.get("answer_idx", row.get("answer", None)),
                "documents": [],
                "task_type": "mcq",
            }
        )

    return examples
