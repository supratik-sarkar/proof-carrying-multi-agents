#!/usr/bin/env python3
"""Renders new synthetic placeholder tables in eight promised rebuttal directories under synthetic_placeholder/."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REBUTTAL_DIR = REPO_ROOT / "artifacts" / "rebuttal"

SYNTHETIC_META = {
    "provenance": "SYNTHETIC_PLACEHOLDER",
    "empirical_status": "NOT_EXECUTED",
    "formal_reporting_allowed": False,
    "server_run_status": "PENDING"
}

LATEX_NOTE = r"\textit{Synthetic placeholder values for internal review; replace after the server run.}"

DIRECTORIES = [
    "table_reconciliation",
    "sv_decomposition",
    "separating_witnesses",
    "citation_only",
    "injection",
    "shift",
    "audit_sampling",
    "backend_manifest"
]

def render_rebuttal_placeholder_tables():
    for d_name in DIRECTORIES:
        target_dir = REBUTTAL_DIR / d_name / "synthetic_placeholder"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Metadata JSON
        meta_out = target_dir / "synthetic_metadata.json"
        meta_out.write_text(json.dumps(SYNTHETIC_META, indent=2), encoding="utf-8")
        
        # 2. README.md
        readme_out = target_dir / "README.md"
        readme_out.write_text(f"""# Synthetic Placeholder Directory - {d_name}

{json.dumps(SYNTHETIC_META, indent=2)}

*Synthetic placeholder values for internal review; replace after the server run.*
""", encoding="utf-8")

        # 3. CSV companion table
        csv_out = target_dir / f"{d_name}_synthetic.csv"
        csv_out.write_text("metric_name,value,provenance\nexample_metric,0.082,SYNTHETIC_PLACEHOLDER\n", encoding="utf-8")

        # 4. LaTeX table source
        tex_content = (
            r"% provenance: SYNTHETIC_PLACEHOLDER" + "\n" +
            r"% empirical_status: NOT_EXECUTED" + "\n" +
            r"% formal_reporting_allowed: false" + "\n" +
            r"% server_run_status: PENDING" + "\n\n" +
            r"\begin{table}[h]" + "\n" +
            r"\centering" + "\n" +
            r"\small" + "\n" +
            rf"{LATEX_NOTE}" + "\n" +
            rf"\caption{{Synthetic placeholder table for {d_name}.}}" + "\n" +
            rf"\label{{tab:rebuttal_preview_{d_name}}}" + "\n" +
            r"\begin{tabular}{lc}" + "\n" +
            r"\hline" + "\n" +
            r"Metric & Value \\" + "\n" +
            r"\hline" + "\n" +
            r"Synthetic Metric & 0.082 \\" + "\n" +
            r"\hline" + "\n" +
            r"\end{tabular}" + "\n" +
            r"\end{table}" + "\n"
        )
        tex_out = target_dir / f"{d_name}_synthetic.tex"
        tex_out.write_text(tex_content, encoding="utf-8")

    print("[REBUTTAL RENDERER] Created synthetic_placeholder/ subdirectories across all 8 rebuttal directories.")

if __name__ == "__main__":
    render_rebuttal_placeholder_tables()
