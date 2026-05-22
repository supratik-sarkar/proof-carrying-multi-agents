from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PAPER_METRICS_PATH = Path("results/tables/csv/paper_metrics.jsonl")


def make_metric_row(
    *,
    run_type: str,
    experiment: str,
    model: str,
    dataset: str,
    seed: int,
    n_examples: int,
    backend: str,
    metrics: dict,
) -> dict:
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type,
        "experiment": experiment,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "n_examples": n_examples,
        "backend": backend,
    }
    row.update(metrics)
    return row


def append_metric_rows(rows: Iterable[dict], path: Path = PAPER_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def overwrite_metric_rows(rows: Iterable[dict], path: Path = PAPER_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
