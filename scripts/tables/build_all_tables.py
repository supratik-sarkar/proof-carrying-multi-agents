from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "results" / "tables" / "csv" / "paper_metrics.jsonl"
TEX_DIR = ROOT / "results" / "tables" / "tex"
GENERATOR = ROOT / "scripts" / "tables" / "make_paper_tables.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PCG-MAS paper LaTeX tables.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow incomplete smoke-test metrics; missing entries are rendered as NA.",
    )
    args = parser.parse_args()

    TEX_DIR.mkdir(parents=True, exist_ok=True)

    validate_cmd = [
        sys.executable,
        "scripts/tables/validate_paper_metrics.py",
        "--rows",
        str(ROWS),
    ]
    if args.allow_partial:
        validate_cmd.append("--allow-partial")

    subprocess.run(validate_cmd, cwd=ROOT, check=True)

    cmd = [
        sys.executable,
        str(GENERATOR),
        "--rows",
        str(ROWS),
        "--outdir",
        str(TEX_DIR),
    ]
    if args.allow_partial:
        cmd.append("--allow-partial")

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    tex_files = sorted(TEX_DIR.glob("*.tex"))
    if not tex_files:
        raise SystemExit(f"Table builder wrote no .tex files to {TEX_DIR}")

    print(f"Wrote {len(tex_files)} LaTeX tables to {TEX_DIR}")
    for p in tex_files:
        print(f"  {p.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
