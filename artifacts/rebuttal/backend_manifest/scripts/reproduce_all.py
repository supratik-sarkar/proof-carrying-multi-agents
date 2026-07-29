#!/usr/bin/env python3
"""Reproduce all backend manifest verification outputs."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from verify_manifest import verify_backend_manifest

def reproduce_all(source_records_path, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bm_res = verify_backend_manifest(source_records_path, out_dir / "backend_manifest_summary.json")
    
    csv_lines = ["total_records,verified_records,required_fields_count,status"]
    csv_lines.append(f"{bm_res['total_records']},{bm_res['verified_records']},{bm_res['required_fields_count']},{bm_res['status']}")
    (out_dir / "backend_manifest_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce backend manifest.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Backend manifest pipeline reproduced successfully.")
