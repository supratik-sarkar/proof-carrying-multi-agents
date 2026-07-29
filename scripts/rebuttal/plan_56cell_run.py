#!/usr/bin/env python3
"""Plan 56-cell execution matrix."""

import argparse
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description="Plan 56-cell execution matrix.")
    parser.add_argument("--plan", action="store_true", help="Print execution plan")
    parser.add_argument("--dry-run", action="store_true", help="Simulate planning check")
    args = parser.parse_args()

    cfg_path = REPO_ROOT / "configs" / "rebuttal_56cell.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models = [m["id"] for m in cfg["matrix"]["models"]]
    datasets = cfg["matrix"]["datasets"]
    total_cells = len(models) * len(datasets)

    print(f"=== 56-CELL EXECUTION PLAN ===")
    print(f"Models ({len(models)}): {models}")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Total Unique Cells: {total_cells}")

    assert total_cells == 56, f"Expected exactly 56 unique cells, got {total_cells}!"
    print("[PASS] 56-cell plan cardinality verified 100%.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
