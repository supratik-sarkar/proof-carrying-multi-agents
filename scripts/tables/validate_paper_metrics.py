from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=Path("results/tables/csv/paper_metrics.jsonl"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    if not args.rows.exists() or args.rows.stat().st_size == 0:
        raise SystemExit(f"Missing metric rows file: {args.rows}")

    rows = [json.loads(x) for x in args.rows.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise SystemExit(f"No metric rows found in {args.rows}")

    if any(row.get("metric_source") == "schema_preflight_stub" for row in rows):
        raise SystemExit(
            f"{args.rows} contains schema_preflight_stub rows. "
            "Do not generate public figures/tables from schema-only preflight rows."
        )

    bad = []
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED_HEADLINE_COLUMNS if row.get(k) is None]
        if missing:
            bad.append((i, missing))

    if bad and not args.allow_partial:
        lines = [
            f"Paper metric validation failed for {args.rows}.",
            "The builders require measured, paper-facing headline columns.",
            "",
        ]
        for i, missing in bad[:10]:
            lines.append(f"row {i} missing: {missing}")
        lines.append("")
        lines.append("This is expected until NoCert/ShieldAgent/AgentRR/PCG-MAS headline metrics are all present.")
        lines.append("Use --allow-partial only for private layout debugging, not for paper/repo artifacts.")
        raise SystemExit("\n".join(lines))

    print(f"Validated {len(rows)} paper metric rows.")
    if bad:
        print(f"Partial validation allowed; {len(bad)} rows have missing headline fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
