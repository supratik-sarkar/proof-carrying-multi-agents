from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "results" / "tables" / "csv" / "paper_metrics.jsonl"
OUT_DIR = ROOT / "results" / "figures"
GENERATOR = ROOT / "scripts" / "figures" / "make_paper_figures.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all PCG-MAS paper figures.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow private layout/debug figure generation when some headline metrics are missing.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GENERATOR.exists():
        raise SystemExit(f"Missing figure generator: {GENERATOR}")

    if not ROWS.exists() or ROWS.stat().st_size == 0:
        raise SystemExit(f"Blocked: no fresh metric rows found at {ROWS}")

    # Strictly validate before the plotting code can fall back to defaults.
    validate_cmd = [sys.executable, "scripts/tables/validate_paper_metrics.py", "--rows", str(ROWS)]
    if args.allow_partial:
        validate_cmd.append("--allow-partial")

    subprocess.run(
        validate_cmd,
        cwd=ROOT,
        check=True,
    )

    before = {p.name for p in OUT_DIR.glob("*.pdf")}

    cmd = [
        sys.executable,
        str(GENERATOR),
        "--rows",
        str(ROWS),
        "--outdir",
        str(OUT_DIR),
    ]
    if args.allow_partial:
        cmd.append("--allow-partial")
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    after = {p.name for p in OUT_DIR.glob("*.pdf")}
    new_files = sorted(after - before)

    if not after:
        raise SystemExit(f"Figure builder wrote no PDFs to {OUT_DIR}")

    print(f"Wrote {len(after)} PDF figures to {OUT_DIR}")
    if new_files:
        print("New PDFs:")
        for name in new_files:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
