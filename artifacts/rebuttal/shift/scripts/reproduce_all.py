#!/usr/bin/env python3
"""Reproduce all shift family evaluation outputs."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from apply_validity_gate import apply_shift_validity_gate

def reproduce_all(source_records_path, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    src_p = Path(source_records_path)
    if not src_p.exists():
        raise FileNotFoundError(f"Source records not found: {src_p}")
        
    lines = src_p.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    if not records:
        raise ValueError("Empty source records file.")
        
    domain_file = script_dir.parent / "source_records" / "shift_family_records.jsonl"
    target_records = str(domain_file) if (src_p.name == "per_example_records.jsonl" and domain_file.exists()) else str(src_p)
    
    sh_res = apply_shift_validity_gate(target_records, out_dir / "shift_validity_summary.json")
    
    csv_lines = ["family_name,evaluated_samples,tpr,tnr,balanced_accuracy,tv_lower_bound,hat_rho,rho_ucb,validity_gate_passed,fallback_action"]
    for fam, m in sh_res["families"].items():
        csv_lines.append(f"{m['family_name']},{m['evaluated_samples']},{m['tpr']:.4f},{m['tnr']:.4f},{m['balanced_accuracy']:.4f},{m['tv_lower_bound']:.4f},{m['hat_rho']:.4f},{m['rho_ucb']:.4f},{m['validity_gate_passed']},{m['fallback_action']}")
        
    (out_dir / "shift_validity_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce shift outputs.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Shift pipeline reproduced successfully.")
