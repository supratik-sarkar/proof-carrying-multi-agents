#!/usr/bin/env python3
"""Reconcile local MacBook and Colab/remote PCG-MAS v4 matrix outputs.

Expected layout:
    results/v4_matrix/local/<dataset>__<model>/
    results/v4_matrix/remote/<dataset>__<model>/

This script does not modify raw experiment files. It creates:
    results/v4_matrix/merged_manifest.json

The merged manifest records local/remote cells, DONE markers, failures,
and JSON outputs so downstream metric extraction can consume both sides.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


R_KEYS = ("r1", "r2", "r3", "r4", "r5")


def as_posix(p: Path) -> str:
    return p.as_posix()


def read_json_safe(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"payload": payload}
    except Exception:
        return None


def file_info(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": as_posix(path),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def scan_cell(cell_dir: Path, source: str) -> dict:
    done = sorted(cell_dir.glob("*.DONE"))
    failed = sorted(cell_dir.glob("*.FAILED.json"))
    jsons = sorted(
        p for p in cell_dir.rglob("*.json")
        if not p.name.endswith(".FAILED.json")
    )

    exp_status = {}
    for r in R_KEYS:
        exp_status[r] = {
            "done": (cell_dir / f"{r}.DONE").exists(),
            "failed": (cell_dir / f"{r}.FAILED.json").exists(),
            "done_marker": as_posix(cell_dir / f"{r}.DONE") if (cell_dir / f"{r}.DONE").exists() else None,
            "failed_marker": as_posix(cell_dir / f"{r}.FAILED.json") if (cell_dir / f"{r}.FAILED.json").exists() else None,
        }

    return {
        "cell": cell_dir.name,
        "source": source,
        "root": as_posix(cell_dir),
        "complete": all(exp_status[r]["done"] for r in R_KEYS),
        "done_count": sum(1 for r in R_KEYS if exp_status[r]["done"]),
        "failed_count": sum(1 for r in R_KEYS if exp_status[r]["failed"]),
        "experiments": exp_status,
        "done_markers": [file_info(p) for p in done],
        "failed_markers": [file_info(p) for p in failed],
        "json_outputs": [file_info(p) for p in jsons],
    }


def scan_root(root: Path, source: str) -> list[dict]:
    if not root.exists():
        return []
    cells = []
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cells.append(scan_cell(cell_dir, source))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, default=Path("results/v4_matrix"))
    parser.add_argument("--output", type=Path, default=Path("results/v4_matrix/merged_manifest.json"))
    args = parser.parse_args()

    local_root = args.matrix_root / "local"
    remote_root = args.matrix_root / "remote"

    local_cells = scan_root(local_root, "local")
    remote_cells = scan_root(remote_root, "remote")

    all_cells = local_cells + remote_cells

    duplicates = {}
    seen = {}
    for cell in all_cells:
        name = cell["cell"]
        if name in seen:
            duplicates.setdefault(name, []).append(cell["source"])
        else:
            seen[name] = cell["source"]

    manifest = {
        "schema": "pcg_mas_v4_merged_matrix_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_root": as_posix(args.matrix_root),
        "local_root": as_posix(local_root),
        "remote_root": as_posix(remote_root),
        "counts": {
            "local_cells": len(local_cells),
            "remote_cells": len(remote_cells),
            "total_cells": len(all_cells),
            "complete_cells": sum(1 for c in all_cells if c["complete"]),
            "partial_cells": sum(1 for c in all_cells if not c["complete"]),
            "failed_cells": sum(1 for c in all_cells if c["failed_count"] > 0),
            "done_markers": sum(c["done_count"] for c in all_cells),
            "failed_markers": sum(c["failed_count"] for c in all_cells),
        },
        "duplicates": duplicates,
        "cells": all_cells,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(json.dumps(manifest["counts"], indent=2))
    if duplicates:
        print("WARNING: duplicate cell names across local/remote:")
        print(json.dumps(duplicates, indent=2))


if __name__ == "__main__":
    main()