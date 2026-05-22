from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pcg.utils.hf_auth import resolve_hf_token


def require_passed(path: Path, label: str, command: str) -> None:
    if not path.exists():
        raise SystemExit(f"Blocked: run {label} first:\n  {command}")

    status = json.loads(path.read_text(encoding="utf-8"))
    if not status.get("passed", False):
        raise SystemExit(f"Blocked: {label} did not pass.")


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full local 40-cell benchmark.")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--allow-full-run", action="store_true")

    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "hf_local", "hf_inference"],
        help="Execution backend. Use hf_inference only when HF token access is available.",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--non-interactive", action="store_true")

    parser.add_argument(
        "--preflight-status",
        type=Path,
        default=Path("artifacts/preflight_2_cells_status.json"),
    )
    parser.add_argument(
        "--preflight40-status",
        type=Path,
        default=Path("artifacts/preflight_40_cells_status.json"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("artifacts/local_40_cells_status.json"),
    )

    args = parser.parse_args()

    if not args.allow_full_run:
        raise SystemExit(
            "Blocked: full local benchmark requires explicit confirmation:\n"
            "  python scripts/runs/run_local_40_cells.py --n 500 --seeds 0 1 2 3 --allow-full-run"
        )

    require_passed(
        args.preflight_status,
        "2-cell preflight",
        "python scripts/runs/run_preflight.py --n 10 --seeds 0",
    )
    require_passed(
        args.preflight40_status,
        "40-cell preflight",
        "python scripts/runs/run_preflight_40_cells.py --n 5 --seeds 0",
    )

    requested_backend = args.backend

    if args.backend == "hf_inference":
        auth = resolve_hf_token(
            explicit_token=args.hf_token,
            require_for_full=True,
            interactive=not args.non_interactive,
        )
        print(auth.message)

        if not auth.full_access:
            print(
                "Remote Hugging Face model access is unavailable. "
                "Switching backend to mock and running the feasible offline path. "
                "This is not a full 7-LLM rerun."
            )
            args.backend = "mock"

    print("Launching local 40-cell run.")
    print(f"n={args.n}, seeds={args.seeds}")
    print(f"requested_backend={requested_backend}")
    print(f"effective_backend={args.backend}")

    # Delegate to the flexible matrix runner. This preserves the option to run
    # arbitrary subsets through scripts/runs/run_matrix.py while keeping this
    # file as the canonical full local 40-cell entrypoint.
    import subprocess
    import sys

    matrix_cmd = [
        sys.executable,
        "scripts/runs/run_matrix.py",
        "--n-examples",
        str(args.n),
        "--seeds",
        *map(str, args.seeds),
        "--experiments",
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
        "--backend",
        args.backend,
        "--allow-full-run",
    ]

    # Full local runs should not silently use dataset fallbacks unless the user
    # selected mock/offline mode after missing HF access.
    if args.backend == "mock":
        matrix_cmd.append("--allow-dataset-fallback")

    print("[delegate]", " ".join(matrix_cmd))
    subprocess.run(matrix_cmd, check=True)

    payload = {
        "passed": True,
        "run_type": "local_40_cells",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_examples": args.n,
        "seeds": args.seeds,
        "cell_count": 40,
        "requested_backend": requested_backend,
        "effective_backend": args.backend,
        "outputs": {
            "figures": "results/figures",
            "tables_csv": "results/tables/csv",
            "tables_tex": "results/tables/tex",
        },
    }

    write_status(args.status, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
