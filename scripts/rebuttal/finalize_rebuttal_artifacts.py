#!/usr/bin/env python3
"""Finalizes rebuttal artifacts from server run records."""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Finalize rebuttal artifacts.")
    parser.add_argument("--server-output-dir", type=Path, default=Path("results/server_runs"))
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    print("=== FINALIZING REBUTTAL ARTIFACTS ===")
    if not args.server_output_dir.exists() and not args.allow_pending:
        print(f"[BLOCKED] Server output directory {args.server_output_dir} does not exist.")
        print("Run 56-cell server execution before finalization.")
        sys.exit(1)

    print("[FINALIZER READY] Artifact finalizer pipeline ready for server run records.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
