#!/usr/bin/env python3
"""Reproduce all audit sampling tables."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from run_sampling_designs import run_all_sampling_designs

def reproduce_all(source_records_path, output_dir):
    src_p = Path(source_records_path)
    if not src_p.exists():
        raise FileNotFoundError(f"Source records file not found: {src_p}")
        
    src_bytes = src_p.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    aud_res = run_all_sampling_designs(source_records_path, out_dir / "audit_sampling_summary.json")
    
    csv_lines = ["design_name,estimate,standard_error,confidence_interval,effective_sample_size,population_size,selection_probability_source"]
    for d, m in aud_res["designs"].items():
        csv_lines.append(f"{d},{m['estimate']:.4f},{m['standard_error']:.4f},{m['confidence_interval']:.4f},{m['effective_sample_size']:.1f},{m['population_size']},{m['selection_probability_source']}")
        
    (out_dir / "audit_sampling_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    
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
        "deterministic_outputs": ["audit_sampling_summary.json", "audit_sampling_summary.csv"]
    }
    (out_dir / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce audit sampling.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Audit sampling pipeline reproduced successfully.")
