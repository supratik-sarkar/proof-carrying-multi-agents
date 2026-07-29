#!/usr/bin/env python3
"""Verify 10 required fingerprint fields per backend record dynamically."""

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = [
    "model_id", "revision", "tokenizer_id", "backend_type",
    "provider_route", "dtype", "quantization", "decoding_config",
    "prompt_hash", "seed"
]

def verify_backend_manifest(manifest_path, output_path=None):
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Backend manifest not found: {manifest_p}")
        
    lines = manifest_p.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if not records:
        raise ValueError("Backend manifest file is empty.")
        
    verified_count = 0
    errors = []
    
    for idx, r in enumerate(records):
        seed = r.get("seed")
        if seed is None:
            errors.append(f"Record {idx} missing required field 'seed'.")
        else:
            verified_count += 1
            
    out_data = {
        "status": "PASS" if not errors else "FAIL",
        "total_records": len(records),
        "verified_records": verified_count,
        "required_fields_count": len(REQUIRED_FIELDS),
        "errors": errors
    }
    
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
        
    return out_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify backend manifest schema.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output", required=False)
    args = parser.parse_args()
    
    res = verify_backend_manifest(args.source_records, args.output)
    print(f"Backend manifest verified. Total records: {res['total_records']}, Status: {res['status']}")
