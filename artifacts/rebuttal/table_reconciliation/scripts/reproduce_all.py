#!/usr/bin/env python3
"""Reproduce all table reconciliation outputs from direct source records."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from reconcile_tables import run_reconciliation

def reproduce_all(source_records_path, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rec_results = run_reconciliation(source_records_path, out_dir / "table_reconciliation_summary.json")
    
    csv_lines = ["cell_id,k_nc,N_nc,k_pcg,N_pcg,h_nc,h_pcg,raw_gain,haldane_anscombe_gain,cov_audit"]
    for cid, m in rec_results["cells"].items():
        csv_lines.append(f"{cid},{m['k_nc']},{m['N_nc']},{m['k_pcg']},{m['N_pcg']},{m['h_nc']:.4f},{m['h_pcg']:.4f},{m['raw_gain']:.4f},{m['haldane_anscombe_gain']:.4f},{m['cov_audit']:.4f}")
        
    (out_dir / "table_reconciliation_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    
    html_content = f"<html><body><h1>Table Reconciliation Report</h1><p>Total Cells: {rec_results['total_cells']}</p></body></html>"
    (out_dir / "table_reconciliation_summary.html").write_text(html_content, encoding="utf-8")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce all table reconciliation outputs.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("Table reconciliation pipeline reproduced successfully.")
