#!/usr/bin/env parser
"""Reproduce injection matrix outputs cleanly from input records."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from run_injection_matrix import run_injection_matrix

def reproduce_all(source_records_path, output_dir):
    src_p = Path(source_records_path)
    if not src_p.exists():
        raise FileNotFoundError(f"Source records file not found: {src_p}")
        
    src_bytes = src_p.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    inj_res = run_injection_matrix(source_records_path, out_dir / "injection_matrix.json")
    
    csv_lines = ["attack_location,verifier_regime,redundancy_k,modelled_attack_success_rate,modelled_detection_rate,modelled_false_refusal_rate"]
    for key, m in inj_res["matrix"].items():
        csv_lines.append(f"{m['attack_location']},{m['verifier_regime']},{m['redundancy_k']},{m['modelled_attack_success_rate']:.4f},{m['modelled_detection_rate']:.4f},{m['modelled_false_refusal_rate']:.4f}")
        
    (out_dir / "injection_matrix.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    
    manifest = {
        "source_records_path": str(src_p),
        "source_records_sha256": src_sha,
        "configuration_paths": [],
        "configuration_sha256": [],
        "script_path": str(Path(__file__)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classification": "MODELLED",
        "empirical_status": "NOT_RUN",
        "generation_timestamp": "2026-07-30T05:00:00Z",
        "deterministic_outputs": ["injection_matrix.json", "injection_matrix.csv"]
    }
    (out_dir / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce injection matrix.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Injection pipeline reproduced successfully.")
