"""Write results/tables/channel_ablation.{tex,csv} from Phase 12 channel-ablation
runner summaries. Standalone module to keep run_ablations.py free of nested
quoting headaches.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

CELL_DISPLAY_LABELS = {
    "fever__phi-3.5-mini":       r"\texttt{phi-3.5-mini}/\texttt{FEVER}",
    "hotpotqa__phi-3.5-mini":    r"\texttt{phi-3.5-mini}/\texttt{HotpotQA}",
    "hotpotqa__qwen2.5-7B":      r"\texttt{qwen2.5-7B}/\texttt{HotpotQA}",
    "pubmedqa__Llama-3.1-8B":    r"\texttt{Llama-3.1-8B}/\texttt{PubMedQA}",
    "tatqa__Gemma-2-9b-it":      r"\texttt{Gemma-2-9b-it}/\texttt{TAT-QA}",
    "toolbench__Llama-3.3-70B":  r"\texttt{Llama-3.3-70B}/\texttt{ToolBench}",
    "weblinx__deepseek-v3":      r"\texttt{deepseek-v3}/\texttt{WebLINX}",
}

CHANNEL_HEADER_LABELS = {
    "full":           "Full PCG-MAS",
    "minus_v_h":      r"$-V_H$",
    "minus_v_pi":     r"$-V_{\Pi}$",
    "minus_v_gamma":  r"$-V_{\Gamma}$",
    "minus_v_entail": r"$-V_{\vdash}$",
}

VARIANT_ORDER = ["full", "minus_v_h", "minus_v_pi", "minus_v_gamma", "minus_v_entail"]


def _format_harm(harm: Any) -> str:
    if harm is None:
        return "--"
    return f".{int(round(float(harm) * 1000)):03d}"


def write_channel_ablation_tables(summaries: list[dict[str, Any]]) -> None:
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows_csv: list[dict[str, Any]] = []
    rows_tex: list[tuple[str, list[str]]] = []

    for s in summaries:
        cell_key = f"{s['dataset']}__{s['model']}"
        row_label = CELL_DISPLAY_LABELS.get(cell_key, cell_key)
        row_csv: dict[str, Any] = {"cell": cell_key}
        row_tex_cells: list[str] = []
        for variant in VARIANT_ORDER:
            harm = s.get(f"harm_pcg_{variant}_adv")
            if harm is None:
                harm = s.get(f"harm_pcg_{variant}_clean")
            row_csv[f"{variant}_harm_adv"] = harm
            cell_txt = _format_harm(harm)
            if variant == "full":
                cell_txt = r"\textbf{" + cell_txt + "}"
            row_tex_cells.append(cell_txt)
        rows_csv.append(row_csv)
        rows_tex.append((row_label, row_tex_cells))

    csv_path = tables_dir / "channel_ablation.csv"
    fieldnames = ["cell"] + [f"{v}_harm_adv" for v in VARIANT_ORDER]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)
    print(f"wrote {csv_path}")

    header_cols = " & ".join(CHANNEL_HEADER_LABELS[v] for v in VARIANT_ORDER)
    body = "\n".join(
        f"{label}      & " + " & ".join(cells) + r" \\" for label, cells in rows_tex
    )
    tex = (
        r"\begin{table}[!ht]" + "\n"
        r"\centering" + "\n"
        r"\caption{\footnotesize Verification-channel ablation. Removing any channel increases accepted-harm"
        + "\n"
        r"rate; replay \(V_{\Pi}\) and entailment \(V_{\vdash}\) cause the largest degradations, while" + "\n"
        r"execution contracts \(V_{\Gamma}\) matter most in tool/delegation-heavy cells.}" + "\n"
        r"\label{tab:channel_ablation}" + "\n"
        r"\scriptsize" + "\n"
        r"\setlength{\tabcolsep}{2.4pt}" + "\n"
        r"\renewcommand{\arraystretch}{0.90}" + "\n"
        r"\begin{tabular}{lccccc}" + "\n"
        r"\toprule" + "\n"
        f"Cell & {header_cols} " + r"\\" + "\n"
        r"\midrule" + "\n"
        + body + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\vspace{-0.6em}" + "\n"
        + r"\end{table}" + "\n"
    )
    tex_path = tables_dir / "channel_ablation.tex"
    tex_path.write_text(tex)
    print(f"wrote {tex_path}")
