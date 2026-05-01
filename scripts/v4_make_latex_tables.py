#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from pcg.eval.v4_constants import MAIN6_CELLS


OUT = Path("docs/tables/v4")


def fmt(x, d=3):
    return f"{float(x):.{d}f}"


def cell_name(r):
    return f"\\texttt{{{r['model']}}} / \\texttt{{{r['dataset']}}}"


def val(r, metric, method):
    return float(r[metric][method])


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(text, encoding="utf-8")
    print(path)


def load():
    p = json.loads(Path("results/v4/proxy_metrics.json").read_text())
    return p["main6_cells"], p["all56_cells"]


def table_main_six(main6):
    rows = []
    for r in main6:
        rows.append(
            f"{cell_name(r)} & "
            f"{fmt(val(r,'harm','no_certificate'))} & "
            f"{fmt(val(r,'harm','shieldagent'))} & "
            f"{fmt(val(r,'harm','pcg_mas'))} & "
            f"{fmt(val(r,'bound_coverage','pcg_mas')/100)} & "
            f"{fmt(val(r,'responsibility_top1','pcg_mas'))} & "
            f"{fmt(val(r,'utility','pcg_mas'))} \\\\"
        )

    text = r"""\begin{table}[t]
\centering
\caption{\small Six-cell headline evaluation across agentic safety, certified audit quality, responsibility attribution, and answer utility. Lower harm is better; higher certified coverage, responsibility accuracy, and utility are better.}
\label{tab:main_six_summary}
\scriptsize
\begin{tabular}{lcccccc}
\toprule
Cell & Harm$_{\mathrm{NoCert}}\downarrow$ & Harm$_{\mathrm{Shield}}\downarrow$ & Harm$_{\mathrm{PCG}}\downarrow$ & Cert. cov.$\uparrow$ & Resp.@1$\uparrow$ & Utility$\uparrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("main_six_summary.tex", text)


def table_r1r4(main6):
    rows = []
    for r in main6:
        audit = val(r, "audit_coverage", "pcg_mas")
        safety_gain = val(r, "harm", "no_certificate") / max(val(r, "harm", "pcg_mas"), 1e-9)
        shield_gap = val(r, "harm", "shieldagent") / max(val(r, "harm", "pcg_mas"), 1e-9)
        resp_lift = val(r, "responsibility_top1", "pcg_mas") - val(r, "responsibility_top1", "shieldagent")
        control_gain = val(r, "harm_weighted_cost", "shieldagent") / max(val(r, "harm_weighted_cost", "pcg_mas"), 1e-9)
        rows.append(
            f"{cell_name(r)} & {fmt(audit)} & {fmt(safety_gain,1)}$\\times$ & "
            f"{fmt(shield_gap,1)}$\\times$ & {fmt(100*resp_lift,1)}pp & {fmt(control_gain,2)}$\\times$ \\\\"
        )

    text = r"""\begin{table}[t]
\centering
\caption{\small R1--R4 consolidated view. Audit coverage summarizes the finite-sample audit envelope; safety gain measures harm reduction relative to no certificate; the Shield-to-PCG gap isolates improvement over SHIELDAGENT; responsibility lift measures replay-diagnostic gain; control gain compares harm-weighted operating cost.}
\label{tab:r1_r4_combined}
\scriptsize
\begin{tabular}{lccccc}
\toprule
Cell & R1 audit cov.$\uparrow$ & R2 safety gain$\uparrow$ & Shield$\rightarrow$PCG gap$\uparrow$ & R3 resp. lift$\uparrow$ & R4 control gain$\uparrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("r1_r4_combined.tex", text)


def table_cost(main6):
    rows = []
    for r in main6:
        rows.append(
            f"{cell_name(r)} & "
            f"{fmt(val(r,'token_multiplier','no_certificate'),2)}$\\times$ & "
            f"{fmt(val(r,'token_multiplier','shieldagent'),2)}$\\times$ & "
            f"{fmt(val(r,'token_multiplier','pcg_mas'),2)}$\\times$ & "
            f"{fmt(val(r,'latency_multiplier','shieldagent'),2)}$\\times$ & "
            f"{fmt(val(r,'latency_multiplier','pcg_mas'),2)}$\\times$ \\\\"
        )

    text = r"""\begin{table}[t]
\centering
\caption{\small Cost overhead on the six highlighted cells. PCG-MAS is more expensive than a policy-only guardrail because it carries replayable evidence certificates, redundancy, and responsibility diagnostics.}
\label{tab:cost_overhead_main}
\scriptsize
\begin{tabular}{lccccc}
\toprule
Cell & NoCert tok. & Shield tok. & PCG tok. & Shield lat. & PCG lat. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("cost_overhead_main.tex", text)


def appendix_remaining(all56, main6):
    main_keys = {(r["model"], r["dataset"]) for r in main6}
    rem = [r for r in all56 if (r["model"], r["dataset"]) not in main_keys]

    rows_summary = []
    rows_r1r4 = []
    rows_cost = []
    for r in rem:
        rows_summary.append(
            f"{cell_name(r)} & {fmt(val(r,'harm','no_certificate'))} & {fmt(val(r,'harm','shieldagent'))} & "
            f"{fmt(val(r,'harm','pcg_mas'))} & {fmt(val(r,'bound_coverage','pcg_mas')/100)} & "
            f"{fmt(val(r,'responsibility_top1','pcg_mas'))} \\\\"
        )
        rows_r1r4.append(
            f"{cell_name(r)} & {fmt(val(r,'audit_coverage','pcg_mas'))} & "
            f"{fmt(val(r,'harm','no_certificate')/max(val(r,'harm','pcg_mas'),1e-9),1)}$\\times$ & "
            f"{fmt(100*(val(r,'responsibility_top1','pcg_mas')-val(r,'responsibility_top1','shieldagent')),1)}pp & "
            f"{fmt(val(r,'harm_weighted_cost','shieldagent')/max(val(r,'harm_weighted_cost','pcg_mas'),1e-9),2)}$\\times$ \\\\"
        )
        rows_cost.append(
            f"{cell_name(r)} & {fmt(val(r,'token_multiplier','no_certificate'),2)}$\\times$ & "
            f"{fmt(val(r,'token_multiplier','shieldagent'),2)}$\\times$ & "
            f"{fmt(val(r,'token_multiplier','pcg_mas'),2)}$\\times$ \\\\"
        )

    write("appendix_remaining_50_summary.tex", r"""\begin{longtable}{lccccc}
\caption{\small Complement to Table~\ref{tab:main_six_summary} over the remaining 50 cells.}\\
\label{tab:appendix_remaining_50_summary}\\
\toprule
Cell & Harm$_{\mathrm{NoCert}}$ & Harm$_{\mathrm{Shield}}$ & Harm$_{\mathrm{PCG}}$ & Cert. cov. & Resp.@1 \\
\midrule
\endfirsthead
\toprule
Cell & Harm$_{\mathrm{NoCert}}$ & Harm$_{\mathrm{Shield}}$ & Harm$_{\mathrm{PCG}}$ & Cert. cov. & Resp.@1 \\
\midrule
\endhead
""" + "\n".join(rows_summary) + r"""
\bottomrule
\end{longtable}
""")

    write("appendix_remaining_50_r1r4.tex", r"""\begin{longtable}{lcccc}
\caption{\small Complement to Table~\ref{tab:r1_r4_combined} over the remaining 50 cells.}\\
\label{tab:appendix_remaining_50_r1r4}\\
\toprule
Cell & Audit cov. & Safety gain & Resp. lift & Control gain \\
\midrule
\endfirsthead
\toprule
Cell & Audit cov. & Safety gain & Resp. lift & Control gain \\
\midrule
\endhead
""" + "\n".join(rows_r1r4) + r"""
\bottomrule
\end{longtable}
""")

    write("appendix_remaining_50_cost.tex", r"""\begin{longtable}{lccc}
\caption{\small Complement to Table~\ref{tab:cost_overhead_main} over the remaining 50 cells.}\\
\label{tab:appendix_remaining_50_cost}\\
\toprule
Cell & No certificate & SHIELDAGENT & PCG-MAS \\
\midrule
\endfirsthead
\toprule
Cell & No certificate & SHIELDAGENT & PCG-MAS \\
\midrule
\endhead
""" + "\n".join(rows_cost) + r"""
\bottomrule
\end{longtable}
""")


def prompt_bank():
    text = r"""\begin{longtable}{p{0.16\linewidth}p{0.26\linewidth}p{0.50\linewidth}}
\caption{\small Prompt and tool-call templates used by the PCG-MAS benchmark. This disclosure mirrors the prompt-template style used by SHIELDAGENT Appendix~H while adapting it to proof-carrying generation, replay, redundancy, intervention, and control.}\\
\label{tab:appendix_prompt_bank}\\
\toprule
Stage & Purpose & Template summary \\
\midrule
\endfirsthead
\toprule
Stage & Purpose & Template summary \\
\midrule
\endhead
Candidate generation & Produce answer candidates and candidate claims & Given prompt, retrieved context, tool outputs, and task metadata, emit atomic claims, answer draft, cited support identifiers, and uncertainty flags. \\
Certificate construction & Build unified proof-carrying certificate & Commit evidence hashes, replay pipeline, schema/tool/memory/policy/delegation metadata, and calibrated confidence into \(Z=(c,S,\Pi,\Gamma,p,\mathrm{meta})\). \\
Unified checker & Verify evidence and execution jointly & Recompute hashes, replay support pipeline, validate schema/tool/policy contract, and run deterministic entailment check before accepting a claim. \\
SHIELDAGENT baseline & Policy-only trajectory shielding baseline & Extract action predicates from the trajectory, retrieve relevant safety-policy rules, assign predicate values, verify rule satisfaction, and emit safety label/explanation. \\
Redundancy selector & Enforce independent support paths & Select \(k\) certificates whose support paths satisfy provenance, tool-overlap, and replayable-overlap separation constraints. \\
Audit channel probe & Estimate finite-sample audit envelope & Probe integrity, replay, checker, and coverage channels to estimate channel-failure upper envelopes used by the audit-decomposition result. \\
Mask-and-replay & Estimate responsibility & Mask evidence/tool/schema/memory/policy/delegation components and replay the same trace to estimate acceptance sensitivity. \\
Risk controller & Choose runtime action & Use certificate-derived risk, utility, token/latency/tool costs, and harm penalty to choose among \texttt{Answer}, \texttt{Verify}, \texttt{Escalate}, and \texttt{Refuse}. \\
Colab reconciliation & Merge remote large-model cells & Normalize remote Llama-3.3-70B and DeepSeek-V3 JSON outputs into the same \(7\times8\) manifest schema used by the local matrix runner. \\
\bottomrule
\end{longtable}
"""
    write("appendix_prompt_bank.tex", text)


def main():
    main6, all56 = load()
    table_main_six(main6)
    table_r1r4(main6)
    table_cost(main6)
    appendix_remaining(all56, main6)
    prompt_bank()


if __name__ == "__main__":
    main()