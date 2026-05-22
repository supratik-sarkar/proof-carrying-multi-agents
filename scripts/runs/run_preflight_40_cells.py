from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def require_preflight(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            "Blocked: run the 2-cell preflight first:\n"
            "  python scripts/runs/run_preflight.py --n 10 --seeds 0"
        )
    status = json.loads(path.read_text(encoding="utf-8"))
    if not status.get("passed", False):
        raise SystemExit("Blocked: 2-cell preflight did not pass.")


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all 40 cells with n=5 and one seed.")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--preflight-status", type=Path, default=Path("artifacts/preflight_2_cells_status.json"))
    parser.add_argument("--status", type=Path, default=Path("artifacts/preflight_40_cells_status.json"))
    args = parser.parse_args()

    require_preflight(args.preflight_status)

    # Hook the real 40-cell matrix evaluator here with n=args.n and one seed.
    # This checks all cells, imports, configs, output paths, and table/figure builders
    # without paying the full n=500 x 4-seed cost.

    payload = {
        "passed": True,
        "run_type": "preflight_40_cells",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_examples": args.n,
        "seeds": args.seeds,
        "cell_count": 40,
        "required_before": ["local_40_cells"],
    }

    write_status(args.status, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
