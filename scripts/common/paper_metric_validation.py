from __future__ import annotations

import json
from pathlib import Path

REQUIRED_HEADLINE_COLUMNS = [
    "model",
    "dataset",
    "coverage",
    "clean_harm_nocert",
    "clean_harm_shieldagent",
    "clean_harm_agentrr",
    "clean_harm_pcg_mas",
    "adv_harm_nocert",
    "adv_harm_shieldagent",
    "adv_harm_agentrr",
    "adv_harm_pcg_mas",
    "responsibility_top1",
    "utility",
    "tokens_nocert",
    "tokens_shieldagent",
    "tokens_agentrr",
    "tokens_pcg_mas",
    "latency_shieldagent",
    "latency_pcg_mas",
]


def read_metric_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Missing metric rows file: {path}")

    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows:
        raise SystemExit(f"No rows found in metric rows file: {path}")

    return rows


def validate_headline_rows(rows: list[dict], *, source: str = "paper_metrics.jsonl", allow_partial: bool = False) -> None:
    if any(row.get("metric_source") == "schema_preflight_stub" for row in rows):
        raise SystemExit(
            f"{source} contains schema_preflight_stub rows. "
            "These are layout/debug rows, not measured paper metrics."
        )

    bad = []
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED_HEADLINE_COLUMNS if row.get(k) is None]
        if missing:
            bad.append((i, missing))

    if bad and not allow_partial:
        lines = [
            f"Paper metric validation failed for {source}.",
            "The figure/table builders require measured, paper-facing columns.",
            "",
        ]
        for i, missing in bad[:10]:
            lines.append(f"row {i} missing: {missing}")
        lines.append("")
        lines.append(
            "Fix collect_paper_metrics.py so raw R1--R5 outputs are pivoted into "
            "paper-facing columns, or run the missing baselines."
        )
        raise SystemExit("\n".join(lines))

    if bad and allow_partial:
        print(f"Partial metric validation allowed for {source}; {len(bad)} rows have missing headline fields.")


def cells_from_rows(rows: list[dict]) -> list[tuple[str, str]]:
    seen = []
    used = set()
    for row in rows:
        model = str(row.get("model"))
        dataset = str(row.get("dataset"))
        key = (model, dataset)
        if model != "None" and dataset != "None" and key not in used:
            used.add(key)
            seen.append(key)
    return seen
