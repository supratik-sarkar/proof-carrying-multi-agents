#!/usr/bin/env python3
"""Reproduce all audit sampling tables."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from run_sampling_designs import run_all_sampling_designs

def reproduce_all(source_records_path, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    src_p = Path(source_records_path)
    if not src_p.exists():
        raise FileNotFoundError(f"Source records not found: {src_p}")
        
    # Open and check file
    lines = src_p.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    if not records:
        raise ValueError("Empty source records file.")
        
    domain_file = script_dir.parent / "source_records" / "audit_draw_records.jsonl"
    target_records = str(domain_file) if (src_p.name == "per_example_records.jsonl" and domain_file.exists()) else str(src_p)
    
    aud_res = run_all_sampling_designs(target_records, out_dir / "audit_sampling_summary.json")
    
    csv_lines = ["design_name,sample_size,effective_sample_size,estimated_harm,interval_width,uncovered_mass_penalty,status"]
    for d, m in aud_res["designs"].items():
        csv_lines.append(f"{m['design_name']},{m['sample_size']},{m['effective_sample_size']},{m['estimated_harm']:.4f},{m['interval_width']:.4f},{m['uncovered_mass_penalty']:.4f},{m['status']}")
        
    (out_dir / "audit_sampling_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce audit sampling.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Audit sampling pipeline reproduced successfully.")
