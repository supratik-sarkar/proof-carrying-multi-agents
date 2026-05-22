from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common.schema import assert_paper_ready, manifest_hash, read_jsonl
from scripts.figures.make_paper_figures import make_all_figures
from scripts.tables.make_paper_tables import make_all_tables


def write_csv_dump(rows: list[dict], outdir: Path) -> None:
    """Write a compact CSV dump of the metric rows used for paper artifacts."""
    import csv

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "paper_metrics.csv"

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PCG-MAS paper figures/tables.")
    parser.add_argument("--metrics", required=True, help="Input JSONL metrics file.")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--tables-csv-dir", default="results/tables/csv")
    parser.add_argument("--tables-tex-dir", default="results/tables/tex")
    parser.add_argument("--manifest", default="artifacts/manifest_hash.txt")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.metrics))
    assert_paper_ready(rows)

    figures_dir = Path(args.figures_dir)
    tables_csv_dir = Path(args.tables_csv_dir)
    tables_tex_dir = Path(args.tables_tex_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_csv_dir.mkdir(parents=True, exist_ok=True)
    tables_tex_dir.mkdir(parents=True, exist_ok=True)

    h = manifest_hash(rows)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(h + "\n", encoding="utf-8")

    write_csv_dump(rows, tables_csv_dir)
    make_all_tables(rows, tables_tex_dir)
    make_all_figures(rows, figures_dir)

    summary = {
        "metrics": args.metrics,
        "manifest_hash": h,
        "figures_dir": str(figures_dir),
        "tables_csv_dir": str(tables_csv_dir),
        "tables_tex_dir": str(tables_tex_dir),
    }
    Path("artifacts/paper_artifact_build_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
