#!/usr/bin/env python3
"""Reproduce all separating witness outputs."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from run_witness_suite import evaluate_witness_suite

def reproduce_all(source_records_path, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    wit_res = evaluate_witness_suite(source_records_path, out_dir / "separating_witnesses.json")
    
    csv_lines = ["witness_id,failed_channel,failed_channels_count,certificate_valid"]
    for w in wit_res["witnesses"]:
        csv_lines.append(f"{w['witness_id']},{w['failed_channel']},{w['failed_channels_count']},{w['certificate_valid']}")
        
    (out_dir / "separating_witnesses.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce witness suite.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Separating witness pipeline reproduced successfully.")
