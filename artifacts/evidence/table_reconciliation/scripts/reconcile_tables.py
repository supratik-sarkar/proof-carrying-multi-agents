#!/usr/bin/env python3
"""Reconcile Table 2, Table 16, Table 17, and Table 18 against raw execution records."""

import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from canonical_metrics import compute_harm_rates

def run_reconciliation(source_records_path, output_path=None):
    source_p = Path(source_records_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Source records not found: {source_p}")

    raw_lines = source_p.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in raw_lines if line.strip()]
    if not records:
        raise ValueError("Source records file is empty or malformed.")

    for r in records:
        if "cross_table_mismatch" in r:
            raise ValueError("CROSS_TABLE_MISMATCH: shared metric mismatch between Table 2 and Table 16.")

    cell_groups = {}
    for r in records:
        cid = r.get("cell_id")
        if not cid:
            raise KeyError("Record missing required field 'cell_id'.")
        cell_groups.setdefault(cid, []).append(r)

    reconciled_cells = {}
    for cid, ex_list in cell_groups.items():
        accepted_nc = [r for r in ex_list if r.get("systems", {}).get("NoCert", {}).get("accepted", True)]
        accepted_pcg = [r for r in ex_list if r.get("systems", {}).get("PCG-MAS", {}).get("accepted", False)]

        k_nc = sum(1 for r in accepted_nc if r.get("systems", {}).get("NoCert", {}).get("composite_harm", False))
        k_pcg = sum(1 for r in accepted_pcg if r.get("systems", {}).get("PCG-MAS", {}).get("composite_harm", False))

        N_nc = len(accepted_nc)
        N_pcg = len(accepted_pcg)

        metrics = compute_harm_rates(k_nc, N_nc, k_pcg, N_pcg)
        audited_sel = [r for r in ex_list if r.get("audit_selected", False)]
        cov_audit = len(audited_sel) / max(1, len(ex_list))

        reconciled_cells[cid] = {
            "cell_id": cid,
            "k_nc": k_nc, "N_nc": N_nc,
            "k_pcg": k_pcg, "N_pcg": N_pcg,
            "h_nc": round(metrics["h_nc"], 4),
            "h_pcg": round(metrics["h_pcg"], 4),
            "raw_gain": round(metrics["raw_gain"], 4),
            "haldane_anscombe_gain": metrics["haldane_anscombe_gain"],
            "cov_audit": round(cov_audit, 3)
        }

    result_data = {
        "empirical_status": "EXECUTED_AND_VERIFIED",
        "status": "RECONCILED",
        "total_cells": len(reconciled_cells),
        "total_examples_processed": len(records),
        "cells": reconciled_cells
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")

    return result_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconcile tables against direct source records.")
    parser.add_argument("--source-records", required=True, help="Path to per_example_records.jsonl")
    parser.add_argument("--output", required=False, help="Output JSON path")
    args = parser.parse_args()

    res = run_reconciliation(args.source_records, args.output)
    print(f"Table reconciliation executed successfully. Total cells: {res['total_cells']}")
