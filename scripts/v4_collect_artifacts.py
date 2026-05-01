#!/usr/bin/env python3
"""Collect PCG-MAS v4 artifacts into one manifest.

This script is intentionally non-invasive. It scans results/ and figures/
for R1-R5 outputs and writes a manifest that the paper/Streamlit app can use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


R_KEYS = ("r1", "r2", "r3", "r4", "r5")


def scan_files(root: Path, suffixes: tuple[str, ...]) -> list[str]:
    if not root.exists():
        return []
    out: list[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(str(p))
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("docs/tables"))
    parser.add_argument("--output", type=Path, default=Path("results/v4/artifact_manifest.json"))
    args = parser.parse_args()

    manifest: dict[str, object] = {
        "results_dir": str(args.results_dir),
        "figures_dir": str(args.figures_dir),
        "tables_dir": str(args.tables_dir),
        "experiments": {},
    }

    all_result_files = scan_files(args.results_dir, (".json", ".jsonl", ".csv"))
    all_figure_files = scan_files(args.figures_dir, (".pdf", ".png", ".svg"))
    all_table_files = scan_files(args.tables_dir, (".tex", ".csv", ".json"))

    for key in R_KEYS:
        manifest["experiments"][key] = {
            "results": [p for p in all_result_files if f"/{key}" in p.lower() or key in Path(p).name.lower()],
            "figures": [p for p in all_figure_files if f"/{key}" in p.lower() or key in Path(p).name.lower()],
            "tables": [p for p in all_table_files if f"/{key}" in p.lower() or key in Path(p).name.lower()],
        }

    manifest["intro"] = {
        "figures": [
            p for p in all_figure_files
            if "intro_hero_v4" in Path(p).name.lower()
        ]
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {args.output}")
    for key in ("intro", *R_KEYS):
        block = manifest["intro"] if key == "intro" else manifest["experiments"][key]
        print(f"{key}: {len(block.get('figures', []))} figures")


if __name__ == "__main__":
    main()