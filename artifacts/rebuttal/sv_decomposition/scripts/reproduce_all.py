#!/usr/bin/env python3
"""Reproduce all S/V decomposition outputs from direct source records."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from compute_sv import compute_sv_metrics
from paired_bootstrap import run_paired_bootstrap

def reproduce_all(source_records_path, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sv_res = compute_sv_metrics(source_records_path, out_dir / "sv_decomposition.json")
    boot_res = run_paired_bootstrap(source_records_path, n_bootstraps=1000, output_path=out_dir / "sv_bootstrap_ci.json")
    
    csv_lines = ["cell_id,N_all,N_pcg_answered,I_nc_all,H_nc_A,H_pcg_A,S,V,S_plus_V,identity_residual"]
    for cid, m in sv_res["cells"].items():
        csv_lines.append(f"{cid},{m['N_all']},{m['N_pcg_answered']},{m['I_nc_all']:.4f},{m['H_nc_A']:.4f},{m['H_pcg_A']:.4f},{m['S']:.4f},{m['V']:.4f},{m['S_plus_V']:.4f},{m['identity_residual']:.14e}")
        
    (out_dir / "sv_decomposition.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce S/V outputs.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    reproduce_all(args.source_records, args.output_dir)
    print("S/V pipeline reproduced successfully.")
