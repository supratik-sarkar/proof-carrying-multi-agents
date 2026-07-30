#!/usr/bin/env python3
"""Refreshes all 18 manuscript tables with unique exact labels derived from synthetic per-example records."""

import json
import os
import shutil
from pathlib import Path
from canonical_metrics import (
    compute_harm_nocert,
    compute_harm_pcg,
    compute_control_coverage,
    compute_audit_coverage,
    compute_safety_gain
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "results" / "tables"

# Clean out old staged/duplicate tex files in OUT_DIR to ensure exact 1-to-1 label mapping
for old_tex in OUT_DIR.rglob("*.tex"):
    try:
        old_tex.unlink()
    except Exception:
        pass

LATEX_NOTE = r"\textit{Synthetic placeholder values for internal review; replace after the server run.}"
FILE_HEADER = (
    r"% provenance: SYNTHETIC_PLACEHOLDER" + "\n" +
    r"% empirical_status: NOT_EXECUTED" + "\n" +
    r"% formal_reporting_allowed: false" + "\n" +
    r"% server_run_status: PENDING" + "\n\n"
)

def render_all(records_path: Path):
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    cells = {}
    for r in records:
        key = (r["model"], r["dataset"])
        if key not in cells:
            cells[key] = []
        cells[key].append(r)

    headline_keys = [
        ("phi-3.5-mini", "fever"),
        ("gemma-2-9b-it", "tatqa"),
        ("llama-3.3-70b", "toolbench"),
        ("deepseek-v3", "weblinx"),
        ("qwen2.5-7b", "hotpotqa"),
        ("llama-3.1-8b", "pubmedqa")
    ]

    all_keys = sorted(cells.keys())
    remaining_50_keys = [k for k in all_keys if k not in headline_keys]

    def make_table_tex(label, caption, keys, note=LATEX_NOTE):
        lines = [
            FILE_HEADER,
            r"\begin{table}[h]",
            r"\centering",
            r"\small",
            f"  {note}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{l|cc|cc|c}",
            r"\hline",
            r"Model / Dataset & $H_{\mathrm{nocert}}$ & $H_{\mathrm{pcg}}$ & $\mathrm{Cov}_{\mathrm{control}}$ & $\mathrm{Cov}_{\mathrm{audit}}$ & $G_{\mathrm{safe}}$ \\",
            r"\hline"
        ]
        for m, d in keys:
            c_recs = cells.get((m, d), [])
            h_nc = compute_harm_nocert(c_recs)
            h_pcg = compute_harm_pcg(c_recs)
            cov_ctrl = compute_control_coverage(c_recs)
            cov_audit = compute_audit_coverage(c_recs)
            gain = compute_safety_gain(h_nc, h_pcg)
            lines.append(f"{m} / {d} & {h_nc:.3f} & {h_pcg:.3f} & {cov_ctrl:.3f} & {cov_audit:.3f} & {gain:.2f}x \\\\")
        lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    # 18 Table Definitions
    tables = [
        ("tab:main_six_summary", "Table 1: Main six summary", headline_keys, "table1_main_six_summary.tex"),
        ("tab:r1_r4_combined_old", "Table 2: R1-R4 combined (submitted)", headline_keys, "table2_r1_r4_combined_old.tex"),
        ("tab:cost_overhead_main", "Table 3: Cost and latency overhead", headline_keys, "table3_cost_overhead_main.tex"),
        ("tab:audit_calibration_summary", "Table 4: Audit calibration summary", headline_keys, "table4_audit_calibration_summary.tex"),
        ("tab:ablations", "Table 5: Ablations summary", headline_keys, "table5_ablations.tex"),
        ("tab:channel_ablation", "Table 6: Channel ablation summary", headline_keys, "table6_channel_ablation.tex"),
        ("tab:appendix_sota_pivot_full", "Table 7: Full SOTA pivot comparison", headline_keys, "table7_appendix_sota_pivot_full.tex"),
        ("tab:responsibility_detail", "Table 8: Responsibility detail", headline_keys, "table8_responsibility_detail.tex"),
        ("tab:r3_open_mixed", "Table 9: Open and mixed conditions", headline_keys, "table9_r3_open_mixed.tex"),
        ("tab:r4_privacy", "Table 10: Privacy analysis (MODELLED)", headline_keys, "table10_r4_privacy.tex"),
        ("tab:r5_scaling", "Table 11: Scaling law analysis (MODELLED)", headline_keys, "table11_r5_scaling.tex"),
        ("tab:hyperparams_controls", "Table 12: Hyperparameters & evaluation controls (PROTOCOL)", headline_keys, "table12_hyperparams_controls.tex"),
        ("tab:main_six_summary_old", "Table 13: Main six summary (old)", headline_keys, "table13_main_six_summary_old.tex"),
        ("tab:appendix_remaining_50_summary_1", "Table 14: Remaining 50 summary (Part 1)", remaining_50_keys[:25], "table14_appendix_remaining_50_summary_1.tex"),
        ("tab:appendix_remaining_50_summary_2", "Table 15: Remaining 50 summary (Part 2)", remaining_50_keys[25:], "table15_appendix_remaining_50_summary_2.tex"),
        ("tab:appendix_remaining_50_r1r4_old", "Table 16: Remaining 50 R1-R4 (submitted)", remaining_50_keys, "table16_appendix_remaining_50_r1r4_old.tex"),
        ("tab:r1_r4_combined", "Table 17: R1-R4 combined (corrected)", headline_keys, "table17_r1_r4_combined.tex"),
        ("tab:appendix_remaining_50_r1r4", "Table 18: Remaining 50 R1-R4 (corrected)", remaining_50_keys, "table18_appendix_remaining_50_r1r4.tex"),
    ]

    for label, caption, keys, fname in tables:
        tex_str = make_table_tex(label, caption, keys)
        (OUT_DIR / fname).write_text(tex_str, encoding="utf-8")

    print("[RENDERER] Successfully generated all 18 manuscript table .tex files in results/tables/")

if __name__ == "__main__":
    records_file = REPO_ROOT / "results" / "tables" / "synthetic_placeholder_records.jsonl"
    render_all(records_file)
