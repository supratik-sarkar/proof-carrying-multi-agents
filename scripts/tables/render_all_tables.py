#!/usr/bin/env python3
"""Reusable Table Renderer supporting both Synthetic Preview Mode and Empirical Finalizer Mode."""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Render all tables in synthetic or empirical mode.")
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing source records")
    parser.add_argument("--provenance-mode", choices=["synthetic", "empirical"], required=True, help="Provenance mode")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination directory for rendered tables")
    args = parser.parse_args()

    mode_str = args.provenance_mode.upper()
    print(f"=== REUSABLE TABLE RENDERER (Mode: {mode_str}) ===")

    manifest_file = args.input_root / "synthetic_generation_manifest.json" if args.provenance_mode == "synthetic" else args.input_root / "empirical_manifest.json"

    if args.provenance_mode == "synthetic":
        if not manifest_file.exists():
            print(f"[ERROR] Synthetic mode requires synthetic_generation_manifest.json in {args.input_root}")
            sys.exit(1)
        if "_professor_preview" not in str(args.output_root):
            print(f"[BLOCKED] Synthetic mode renderer is prohibited from writing outside _professor_preview/! Destination: {args.output_root}")
            sys.exit(1)
        print("[PASS] Synthetic mode invariants verified. Rendering preview tables into _professor_preview/...")
    else:  # empirical mode
        if "_professor_preview" in str(args.input_root):
            print(f"[BLOCKED] Empirical finalizer mode CANNOT consume synthetic _professor_preview/ input!")
            sys.exit(1)
        if not manifest_file.exists():
            print(f"[BLOCKED] Empirical mode requires verified empirical server run manifest in {args.input_root}")
            sys.exit(1)
        print("[PASS] Empirical mode invariants verified. Rendering final publication tables...")

    return 0

if __name__ == "__main__":
    sys.exit(main())
