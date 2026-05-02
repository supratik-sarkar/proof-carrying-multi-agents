#!/usr/bin/env python3
"""Collect PCG-MAS v4 artifacts into a single manifest.

This script scans the v4 artifact layout after running:

    scripts/v4_make_proxy_metrics.py
    src/pcg/eval/intro_hero_v4.py
    scripts/v4_make_r1_r5_figures.py
    scripts/v4_make_latex_tables.py

It records:
    - intro hero figures
    - R1--R5 v4 figures
    - v4 LaTeX tables
    - metric JSON files
    - local/remote/merged matrix result locations

The manifest is intentionally lightweight and does not validate numerical
correctness. It only records what artifacts exist and where they live.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


R_KEYS = ("r1", "r2", "r3", "r4", "r5")

V4_FIGURE_PATTERNS = {
    "r1": ("r1_audit_decomposition_v4",),
    "r2": ("r2_redundancy_surface_v4",),
    "r3": ("r3_responsibility_v4",),
    "r4": ("r4_control_frontier_v4",),
    "r5": ("r5_overhead_v4",),
}

V4_TABLE_FILES = {
    "main": (
        "main_six_summary.tex",
        "r1_r4_combined.tex",
        "cost_overhead_main.tex",
    ),
    "appendix": (
        "appendix_remaining_50_summary.tex",
        "appendix_remaining_50_r1r4.tex",
        "appendix_remaining_50_cost.tex",
        "appendix_prompt_bank.tex",
    ),
}


def as_posix(path: Path) -> str:
    return path.as_posix()


def scan_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )


def file_info(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": as_posix(path),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def collect_matching(files: Iterable[Path], needles: Iterable[str]) -> list[dict[str, object]]:
    needles_l = tuple(n.lower() for n in needles)
    out: list[dict[str, object]] = []
    for p in files:
        name = p.name.lower()
        full = as_posix(p).lower()
        if any(n in name or n in full for n in needles_l):
            out.append(file_info(p))
    return out


def collect_exact(files: Iterable[Path], names: Iterable[str]) -> list[dict[str, object]]:
    wanted = set(names)
    return [file_info(p) for p in files if p.name in wanted]


def collect_existing(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [file_info(p) for p in paths if p.exists() and p.is_file()]


def collect_done_markers(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    return [file_info(p) for p in sorted(root.rglob("*.DONE"))]


def collect_failed_markers(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    return [file_info(p) for p in sorted(root.rglob("*.FAILED.json"))]


def collect_matrix_json(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("*.json")):
        if p.name.endswith(".FAILED.json"):
            continue
        out.append(file_info(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("docs/tables/v4"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--output", type=Path, default=Path("results/v4/artifact_manifest.json"))
    args = parser.parse_args()

    v4_results_dir = args.results_dir / "v4"
    local_matrix_dir = args.results_dir / "v4_matrix" / "local"
    remote_matrix_dir = args.results_dir / "v4_matrix" / "remote"

    all_result_files = scan_files(args.results_dir, (".json", ".jsonl", ".csv"))
    all_figure_files = scan_files(args.figures_dir, (".pdf", ".png", ".svg"))
    all_table_files = scan_files(args.tables_dir, (".tex", ".csv", ".json"))
    all_doc_files = scan_files(args.docs_dir, (".png", ".pdf", ".svg", ".md"))

    intro_figures = collect_matching(all_figure_files, ("intro_hero_v4",))
    readme_figures = collect_existing([args.docs_dir / "intro_hero_v4.png"])

    experiments: dict[str, object] = {}
    for key in R_KEYS:
        experiments[key] = {
            "figures": collect_matching(all_figure_files, V4_FIGURE_PATTERNS[key]),
            "legacy_or_raw_results": [
                file_info(p) for p in all_result_files
                if f"/{key}/" in as_posix(p).lower()
                or p.name.lower() == f"{key}.json"
                or p.name.lower().startswith(f"{key}_")
            ],
            "tables": [
                file_info(p) for p in all_table_files
                if key in p.name.lower()
            ],
        }

    table_manifest = {
        "main_text": collect_exact(all_table_files, V4_TABLE_FILES["main"]),
        "appendix": collect_exact(all_table_files, V4_TABLE_FILES["appendix"]),
        "all": [file_info(p) for p in all_table_files],
    }

    metric_files = collect_existing([
        v4_results_dir / "proxy_metrics.json",
        v4_results_dir / "intro_hero_metrics.json",
    ])

    matrix_manifest = {
        "local": {
            "root": as_posix(local_matrix_dir),
            "done_markers": collect_done_markers(local_matrix_dir),
            "failed_markers": collect_failed_markers(local_matrix_dir),
            "json_outputs": collect_matrix_json(local_matrix_dir),
        },
        "remote": {
            "root": as_posix(remote_matrix_dir),
            "done_markers": collect_done_markers(remote_matrix_dir),
            "failed_markers": collect_failed_markers(remote_matrix_dir),
            "json_outputs": collect_matrix_json(remote_matrix_dir),
        },
        "merged_manifest": collect_existing([
            args.results_dir / "v4_matrix" / "merged_manifest.json",
        ]),
    }

    manifest: dict[str, object] = {
        "schema": "pcg_mas_v4_artifact_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {
            "results_dir": as_posix(args.results_dir),
            "figures_dir": as_posix(args.figures_dir),
            "tables_dir": as_posix(args.tables_dir),
            "docs_dir": as_posix(args.docs_dir),
        },
        "intro": {
            "figures": intro_figures,
            "readme_figures": readme_figures,
        },
        "experiments": experiments,
        "tables": table_manifest,
        "metrics": metric_files,
        "matrix": matrix_manifest,
        "counts": {
            "intro_figures": len(intro_figures),
            "readme_figures": len(readme_figures),
            "r_figures_total": sum(
                len(experiments[k]["figures"])  # type: ignore[index]
                for k in R_KEYS
            ),
            "main_tables": len(table_manifest["main_text"]),
            "appendix_tables": len(table_manifest["appendix"]),
            "all_tables": len(table_manifest["all"]),
            "metric_files": len(metric_files),
            "local_done_markers": len(matrix_manifest["local"]["done_markers"]),  # type: ignore[index]
            "remote_done_markers": len(matrix_manifest["remote"]["done_markers"]),  # type: ignore[index]
            "local_failed_markers": len(matrix_manifest["local"]["failed_markers"]),  # type: ignore[index]
            "remote_failed_markers": len(matrix_manifest["remote"]["failed_markers"]),  # type: ignore[index]
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {args.output}")
    print(f"intro figures       : {manifest['counts']['intro_figures']}")
    print(f"README figures      : {manifest['counts']['readme_figures']}")
    for key in R_KEYS:
        print(f"{key} figures         : {len(experiments[key]['figures'])}")  # type: ignore[index]
    print(f"main tables         : {manifest['counts']['main_tables']}")
    print(f"appendix tables     : {manifest['counts']['appendix_tables']}")
    print(f"metric files        : {manifest['counts']['metric_files']}")
    print(f"local DONE markers  : {manifest['counts']['local_done_markers']}")
    print(f"remote DONE markers : {manifest['counts']['remote_done_markers']}")


if __name__ == "__main__":
    main()