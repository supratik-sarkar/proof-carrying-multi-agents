#!/usr/bin/env python3
"""Reproduce all citation-only comparative benchmark outputs."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from match_coverage import compute_citation_comparisons

def reproduce_all(source_records_path, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cit_res = compute_citation_comparisons(source_records_path, out_dir / "citation_only_comparison.json")
    
    csv_lines = ["system_name,evaluated_examples,accepted_count,h_support,h_exec,h_composite,coverage,utility,tokens,latency_ms"]
    for sys_name, m in cit_res["systems"].items():
        csv_lines.append(f"{sys_name},{m['evaluated_examples']},{m['accepted_count']},{m['h_support']:.4f},{m['h_exec']:.4f},{m['h_composite']:.4f},{m['coverage']:.4f},{m['utility']:.4f},{m['tokens']},{m['latency_ms']}")
        
    (out_dir / "citation_only_comparison.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce citation-only outputs.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Citation-only comparative pipeline reproduced successfully.")
