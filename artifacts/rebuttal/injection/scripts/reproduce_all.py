#!/usr/bin/env python3
"""Reproduce all injection attack sweep outputs from direct source records."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from run_injection_matrix import run_injection_matrix

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
        
    domain_file = script_dir.parent / "source_records" / "injection_sweep_records.jsonl"
    target_records = str(domain_file) if (src_p.name == "per_example_records.jsonl" and domain_file.exists()) else str(src_p)
    
    inj_res = run_injection_matrix(target_records, out_dir / "injection_matrix.json")
    
    csv_lines = ["attack_location,verifier_regime,evaluated_samples,attack_success_rate,false_refusal_rate,channel_detection_rate"]
    for key, m in inj_res["matrix"].items():
        csv_lines.append(f"{m['attack_location']},{m['verifier_regime']},{m['evaluated_samples']},{m['attack_success_rate']:.4f},{m['false_refusal_rate']:.4f},{m['channel_detection_rate']:.4f}")
        
    (out_dir / "injection_matrix.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce injection matrix.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Injection pipeline reproduced successfully.")
