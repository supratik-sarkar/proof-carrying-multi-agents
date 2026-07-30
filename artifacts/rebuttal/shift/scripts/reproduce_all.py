#!/usr/bin/env python3
"""Reproduce all shift family evaluation outputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from apply_validity_gate import apply_shift_validity_gate

def reproduce_all(source_records_path, output_dir):
    src_p = Path(source_records_path)
    if not src_p.exists():
        raise FileNotFoundError(f"Source records file not found: {src_p}")

    src_bytes = src_p.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sh_res = apply_shift_validity_gate(source_records_path, out_dir / "shift_validity_summary.json")

    csv_lines = ["family_name,tpr,tnr,balanced_accuracy,tv_lower_bound,hat_rho,rho_ucb,validity_gate_passed,fallback_action"]
    for fam, m in sh_res["families"].items():
        csv_lines.append(f"{m['family_name']},{m['tpr']:.4f},{m['tnr']:.4f},{m['balanced_accuracy']:.4f},{m['tv_lower_bound']:.4f},{m['hat_rho']:.4f},{m['rho_ucb']:.4f},{m['validity_gate_passed']},{m['fallback_action']}")

    (out_dir / "shift_validity_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

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
        "deterministic_outputs": ["shift_validity_summary.json", "shift_validity_summary.csv"]
    }
    (out_dir / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce shift outputs.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    reproduce_all(args.source_records, args.output_dir)
    print("Shift pipeline reproduced successfully.")
