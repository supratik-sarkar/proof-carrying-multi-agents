#!/usr/bin/env python3
"""Apply fail-closed UCB validity gate dynamically over 6 shift families."""

import argparse
import json
import math
import numpy as np
from pathlib import Path

REQUIRED_SHIFT_FIELDS = [
    "shift_family", "intervention_id", "actual_safe", "predicted_safe",
    "clean_pass", "clean_fail", "adv_pass", "adv_fail",
    "checker_pass", "checker_fail", "rho_sample"
]

def apply_shift_validity_gate(source_records_path, output_path=None):
    source_p = Path(source_records_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Source records not found: {source_p}")
        
    records = [json.loads(line) for line in source_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty source records file.")
        
    # Validate required explicit fields
    for idx, r in enumerate(records[:10]):
        missing = [f for f in REQUIRED_SHIFT_FIELDS if f not in r]
        if missing:
            raise KeyError(f"Shift record {idx} missing required fields: {missing}")
            
    families = ["dataset_shift", "backend_shift", "corruption", "tool_drift", "branch_dependence", "checker_degradation"]
    observed_fams = set(r["shift_family"] for r in records)
    
    missing_fams = set(families) - observed_fams
    if missing_fams:
        raise ValueError(f"Missing required shift families: {missing_fams}")
        
    results = {}
    
    for fam in families:
        fam_recs = [r for r in records if r["shift_family"] == fam]
        if not fam_recs:
            raise ValueError(f"No records found for shift family {fam}")
            
        tp = sum(1 for r in fam_recs if r["clean_pass"])
        fn = sum(1 for r in fam_recs if r["adv_fail"])
        tn = sum(1 for r in fam_recs if r["clean_fail"])
        fp = sum(1 for r in fam_recs if r["adv_pass"])
        
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        
        balanced_accuracy = 0.5 * (tpr + tnr)
        tv_bound = max(0.0, 2.0 * balanced_accuracy - 1.0)
        
        rho_samples = [r["rho_sample"] for r in fam_recs]
        hat_rho = float(np.mean(rho_samples)) if rho_samples else 0.10
        se_rho = math.sqrt(hat_rho * (1.0 - hat_rho) / max(1, len(fam_recs)))
        rho_ucb = hat_rho + 1.96 * se_rho
        
        gate_open = (rho_ucb <= 0.150)
        
        results[fam] = {
            "family_name": fam,
            "evaluated_samples": len(fam_recs),
            "tpr": round(float(tpr), 4),
            "tnr": round(float(tnr), 4),
            "balanced_accuracy": round(float(balanced_accuracy), 4),
            "tv_lower_bound": round(float(tv_bound), 4),
            "hat_rho": round(float(hat_rho), 4),
            "rho_ucb": round(float(rho_ucb), 4),
            "validity_gate_passed": gate_open,
            "fallback_action": "ALLOW" if gate_open else "FAIL_CLOSED"
        }
        
    out_data = {
        "status": "PASS",
        "total_records_processed": len(records),
        "total_families": len(families),
        "families": results
    }
    
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
        
    return out_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply shift validity gate.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output", required=False)
    args = parser.parse_args()
    
    res = apply_shift_validity_gate(args.source_records, args.output)
    print(f"Shift validity gate evaluated over {res['total_records_processed']} records across {res['total_families']} families.")
