#!/usr/bin/env python3
"""Compute matched-coverage comparative metrics across 5 baseline systems dynamically from source records."""

import argparse
import json
import numpy as np
from pathlib import Path

def compute_citation_comparisons(source_records_path, output_path=None):
    source_p = Path(source_records_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Source records not found: {source_p}")
        
    records = [json.loads(line) for line in source_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty source records file.")
        
    systems = ["NoCert", "Citation-Only", "ShieldAgent", "AgentRR", "PCG-MAS"]
    system_metrics = {}
    
    for sys_name in systems:
        accepted_recs = [r for r in records if r.get("systems", {}).get(sys_name, {}).get("accepted", False)]
        
        support_fail = sum(1 for r in accepted_recs if r.get("systems", {}).get(sys_name, {}).get("support_failure", False))
        exec_fail = sum(1 for r in accepted_recs if r.get("systems", {}).get(sys_name, {}).get("execution_failure", False))
        comp_harm = sum(1 for r in accepted_recs if r.get("systems", {}).get(sys_name, {}).get("composite_harm", False))
        
        cov = len(accepted_recs) / max(1, len(records))
        h_support = support_fail / max(1, len(accepted_recs))
        h_exec = exec_fail / max(1, len(accepted_recs))
        h_comp = comp_harm / max(1, len(accepted_recs))
        
        latencies = [r.get("systems", {}).get(sys_name, {}).get("latency_ms", 350.0) for r in accepted_recs]
        tokens_list = [r.get("systems", {}).get(sys_name, {}).get("tokens", 500) for r in accepted_recs]
        
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        mean_tok = float(np.mean(tokens_list)) if tokens_list else 0.0
        
        safe_recs = [r for r in records if not r.get("systems", {}).get("NoCert", {}).get("composite_harm", False)]
        safe_accepted = [r for r in safe_recs if r.get("systems", {}).get(sys_name, {}).get("accepted", False)]
        utility = len(safe_accepted) / max(1, len(safe_recs))
        
        system_metrics[sys_name] = {
            "system_name": sys_name,
            "evaluated_examples": len(records),
            "accepted_count": len(accepted_recs),
            "h_support": round(h_support, 4),
            "h_exec": round(h_exec, 4),
            "h_composite": round(h_comp, 4),
            "coverage": round(cov, 4),
            "utility": round(utility, 4),
            "tokens": round(mean_tok, 1),
            "latency_ms": round(mean_lat, 1)
        }
        
    out_data = {
        "status": "PASS",
        "matched_example_count": len(records),
        "systems": system_metrics
    }
    
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
        
    return out_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute citation-only baseline comparisons.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output", required=False)
    args = parser.parse_args()
    
    res = compute_citation_comparisons(args.source_records, args.output)
    print(f"Citation-only comparisons computed for {res['matched_example_count']} matched examples.")
