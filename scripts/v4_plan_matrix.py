#!/usr/bin/env python3
"""Print the PCG-MAS v4 experiment matrix.

This does not run models. It tells you exactly which cells are local-Mac
cells and which cells should be run on Colab.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v4_matrix.yaml"))
    parser.add_argument("--mode", choices=["local", "remote", "all"], default="local")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    datasets = cfg["datasets"]
    backends = cfg["backends"]

    rows: list[tuple[str, str, str]] = []
    for ds in datasets:
        for be in backends:
            is_local = bool(ds.get("local", True)) and bool(be.get("local", True))
            if args.mode == "local" and not is_local:
                continue
            if args.mode == "remote" and is_local:
                continue
            rows.append((ds["name"], be["name"], "local" if is_local else "remote"))

    print(f"Config: {args.config}")
    print(f"Mode  : {args.mode}")
    print(f"Cells : {len(rows)}")
    print()
    for i, (ds, be, mode) in enumerate(rows, start=1):
        print(f"{i:02d}. dataset={ds:<22} backend={be:<24} mode={mode}")


if __name__ == "__main__":
    main()