#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


OUT_DIRS = [Path("docs/tables/v4")]


# ---------------------------------------------------------------------
# Shared LaTeX helpers
# ---------------------------------------------------------------------

def fmt(x, d: int = 1) -> str:
    return f"{float(x):.{d}f}"


def fmt_pct(x, d: int = 1) -> str:
    return f"{100.0 * float(x):.{d}f}"


def tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def tt(s: str) -> str:
    return rf"\texttt{{{tex_escape(s)}}}"


def cell_name(r: dict) -> str:
    return rf"{tt(r['model'])} / {tt(r['dataset'])}"


def val(r: dict, metric: str, method: str) -> float:
    return float(r[metric][method])


def best(x: str) -> str:
    return rf"\textbf{{{x}}}"


def write_all(name: str, text: str) -> None:
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        path.write_text(text.strip() + "\n", encoding="utf-8")
        print(path)


def load() -> tuple[list[dict], list[dict]]:
    p = json.loads(Path("results/v4/proxy_metrics.json").read_text(encoding="utf-8"))
    return p["main6_cells"], p["all56_cells"]


def table_star(
    *,
    label: str,
    caption: str,
    tabular: str,
    size: str = r"\scriptsize",
    stretch: str = "1.05",
    placement: str = "t",
) -> str:
    return rf"""
\begin{{table*}}[{placement}]
\centering
{size}
\renewcommand{{\arraystretch}}{{{stretch}}}
\caption{{{caption}}}
\label{{{label}}}
{tabular}
\end{{table*}}
"""


def longtable(
    *,
    label: str,
    caption: str,
    colspec: str,
    header: Sequence[str],
    rows: Sequence[str],
    size: str = r"\scriptsize",
    stretch: str = "1.04",
) -> str:
    h = "\n".join(header)
    body = "\n".join(rows)
    return rf"""
{size}
\renewcommand{{\arraystretch}}{{{stretch}}}
\begin{{longtable}}{{{colspec}}}
\caption{{{caption}}}
\label{{{label}}}\\
\toprule
{h}
\midrule
\endfirsthead
\toprule
{h}
\midrule
\endhead
{body}
\bottomrule
\end{{longtable}}
\normalsize
"""


# ---------------------------------------------------------------------
# Main-text Table 1: six-cell headline
# ---------------------------------------------------------------------

def table_main_six(main6: list[dict]) -> None:
    rows = []
    for r in main6:
        h_no = val(r, "harm", "no_certificate")
        h_sh = val(r, "harm", "shieldagent")
        h_pcg = val(r, "harm", "pcg_mas")
        h_best = min(h_no, h_sh, h_pcg)

        no_s = best(fmt_pct(h_no)) if h_no == h_best else fmt_pct(h_no)
        sh_s = best(fmt_pct(h_sh)) if h_sh == h_best else fmt_pct(h_sh)
        pcg_s = best(fmt_pct(h_pcg)) if h_pcg == h_best else fmt_pct(h_pcg)

        cert_cov = val(r, "bound_coverage", "pcg_mas")
        # bound_coverage is stored as 0--100 in your proxy metrics.
        cert_cov_s = fmt(cert_cov, 1) if cert_cov > 1.0 else fmt_pct(cert_cov)

        resp = val(r, "responsibility_top1", "pcg_mas")
        util = val(r, "utility", "pcg_mas")
        safety_gain = h_no / max(h_pcg, 1e-9)

        rows.append(
            rf"{cell_name(r)} & "
            rf"{no_s} & {sh_s} & {pcg_s} & "
            rf"{best(cert_cov_s)} & {best(fmt_pct(resp))} & {best(fmt_pct(util))} & "
            rf"{best(fmt(safety_gain, 1) + r'$\times$')} \\"
        )

    tab = r"""\begin{tabular}{l|ccc|ccc|c}
\toprule
\multirow{2}{*}{\textbf{Cell}} &
\multicolumn{3}{c|}{\textbf{Bad accepted claim rate} $\downarrow$} &
\multicolumn{3}{c|}{\textbf{PCG-MAS diagnostics} $\uparrow$} &
\multirow{2}{*}{\textbf{Safety gain} $\uparrow$} \\
& No cert. & SHIELD & PCG-MAS & Cert. cov. & Resp.@1 & Utility & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""

    text = table_star(
        label="tab:main_six_summary",
        caption=(
            r"\textbf{Six-cell headline evaluation.} "
            r"Bad accepted claim rate is reported in percent; lower is better. "
            r"Certified coverage, responsibility@1, and utility are reported in percent; higher is better. "
            r"The best safety value in each row is bold."
        ),
        tabular=tab,
    )
    write_all("main_six_summary.tex", text)


# ---------------------------------------------------------------------
# Main-text Table 2: R1--R4 combined
# ---------------------------------------------------------------------

def table_r1r4(main6: list[dict]) -> None:
    rows = []
    for r in main6:
        audit = val(r, "audit_coverage", "pcg_mas")
        audit_s = fmt(audit, 1) if audit > 1.0 else fmt_pct(audit)

        h_no = val(r, "harm", "no_certificate")
        h_sh = val(r, "harm", "shieldagent")
        h_pcg = val(r, "harm", "pcg_mas")

        safety_gain = h_no / max(h_pcg, 1e-9)
        shield_gap = h_sh / max(h_pcg, 1e-9)
        resp_lift = 100.0 * (
            val(r, "responsibility_top1", "pcg_mas")
            - val(r, "responsibility_top1", "shieldagent")
        )
        control_gain = (
            val(r, "harm_weighted_cost", "shieldagent")
            / max(val(r, "harm_weighted_cost", "pcg_mas"), 1e-9)
        )

        rows.append(
            rf"{cell_name(r)} & "
            rf"{best(audit_s)} & "
            rf"{best(fmt(safety_gain, 1) + r'$\times$')} & "
            rf"{best(fmt(shield_gap, 1) + r'$\times$')} & "
            rf"{best('+' + fmt(resp_lift, 1) + ' pp')} & "
            rf"{best(fmt(control_gain, 2) + r'$\times$')} \\"
        )

    tab = r"""\begin{tabular}{l|cc|ccc}
\toprule
\multirow{2}{*}{\textbf{Cell}} &
\multicolumn{2}{c|}{\textbf{Audit and redundancy}} &
\multicolumn{3}{c}{\textbf{Diagnosis and control}} \\
& R1 audit cov. $\uparrow$ & R2 safety gain $\uparrow$ &
Shield$\rightarrow$PCG $\uparrow$ & R3 resp. lift $\uparrow$ & R4 control gain $\uparrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""

    text = table_star(
        label="tab:r1_r4_combined",
        caption=(
            r"\textbf{R1--R4 consolidated diagnostics.} "
            r"Audit coverage summarizes the finite-sample audit envelope; safety gain measures harm reduction "
            r"relative to no certificate; Shield$\rightarrow$PCG isolates improvement over SHIELDAGENT; "
            r"responsibility lift measures replay-diagnostic gain; control gain compares harm-weighted operating cost."
        ),
        tabular=tab,
    )
    write_all("r1_r4_combined.tex", text)


# ---------------------------------------------------------------------
# Main-text Table 3: overhead
# ---------------------------------------------------------------------

def table_cost(main6: list[dict]) -> None:
    rows = []
    for r in main6:
        tok_no = val(r, "token_multiplier", "no_certificate")
        tok_sh = val(r, "token_multiplier", "shieldagent")
        tok_pcg = val(r, "token_multiplier", "pcg_mas")
        lat_sh = val(r, "latency_multiplier", "shieldagent")
        lat_pcg = val(r, "latency_multiplier", "pcg_mas")

        h_no = val(r, "harm", "no_certificate")
        h_pcg = val(r, "harm", "pcg_mas")
        safety_gain = h_no / max(h_pcg, 1e-9)
        cost_per_gain = tok_pcg / max(safety_gain, 1e-9)

        rows.append(
            rf"{cell_name(r)} & "
            rf"{best(fmt(tok_no, 2) + r'$\times$')} & "
            rf"{fmt(tok_sh, 2)}$\times$ & "
            rf"{fmt(tok_pcg, 2)}$\times$ & "
            rf"{fmt(lat_sh, 2)}$\times$ & "
            rf"{fmt(lat_pcg, 2)}$\times$ & "
            rf"{best(fmt(safety_gain, 1) + r'$\times$')} & "
            rf"{fmt(cost_per_gain, 2)} \\"
        )

    tab = r"""\begin{tabular}{l|ccc|cc|cc}
\toprule
\multirow{2}{*}{\textbf{Cell}} &
\multicolumn{3}{c|}{\textbf{Token multiplier} $\downarrow$} &
\multicolumn{2}{c|}{\textbf{Latency multiplier} $\downarrow$} &
\multicolumn{2}{c}{\textbf{Safety return}} \\
& No cert. & SHIELD & PCG-MAS & SHIELD & PCG-MAS & Harm red. $\uparrow$ & Cost/gain $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""

    text = table_star(
        label="tab:cost_overhead_main",
        caption=(
            r"\textbf{Certification overhead.} "
            r"No certificate is cheapest, SHIELDAGENT adds policy-reasoning cost, and PCG-MAS adds "
            r"certificate, replay, redundancy, and audit overhead. The final columns report the safety return "
            r"obtained for that cost."
        ),
        tabular=tab,
    )
    write_all("cost_overhead_main.tex", text)


# ---------------------------------------------------------------------
# Appendix tables: remaining 50 cells
# ---------------------------------------------------------------------

def appendix_remaining(all56: list[dict], main6: list[dict]) -> None:
    main_keys = {(r["model"], r["dataset"]) for r in main6}
    rem = [r for r in all56 if (r["model"], r["dataset"]) not in main_keys]

    rows_summary = []
    rows_r1r4 = []
    rows_cost = []

    for r in rem:
        h_no = val(r, "harm", "no_certificate")
        h_sh = val(r, "harm", "shieldagent")
        h_pcg = val(r, "harm", "pcg_mas")
        safety_gain = h_no / max(h_pcg, 1e-9)

        cert_cov = val(r, "bound_coverage", "pcg_mas")
        cert_cov_s = fmt(cert_cov, 1) if cert_cov > 1.0 else fmt_pct(cert_cov)

        rows_summary.append(
            rf"{cell_name(r)} & "
            rf"{fmt_pct(h_no)} & {fmt_pct(h_sh)} & {best(fmt_pct(h_pcg))} & "
            rf"{best(cert_cov_s)} & {best(fmt_pct(val(r, 'responsibility_top1', 'pcg_mas')))} & "
            rf"{best(fmt(safety_gain, 1) + r'$\times$')} \\"
        )

        audit = val(r, "audit_coverage", "pcg_mas")
        audit_s = fmt(audit, 1) if audit > 1.0 else fmt_pct(audit)
        resp_lift = 100.0 * (
            val(r, "responsibility_top1", "pcg_mas")
            - val(r, "responsibility_top1", "shieldagent")
        )
        control_gain = (
            val(r, "harm_weighted_cost", "shieldagent")
            / max(val(r, "harm_weighted_cost", "pcg_mas"), 1e-9)
        )

        rows_r1r4.append(
            rf"{cell_name(r)} & "
            rf"{best(audit_s)} & "
            rf"{best(fmt(safety_gain, 1) + r'$\times$')} & "
            rf"{best('+' + fmt(resp_lift, 1) + ' pp')} & "
            rf"{best(fmt(control_gain, 2) + r'$\times$')} \\"
        )

        rows_cost.append(
            rf"{cell_name(r)} & "
            rf"{best(fmt(val(r, 'token_multiplier', 'no_certificate'), 2) + r'$\times$')} & "
            rf"{fmt(val(r, 'token_multiplier', 'shieldagent'), 2)}$\times$ & "
            rf"{fmt(val(r, 'token_multiplier', 'pcg_mas'), 2)}$\times$ & "
            rf"{fmt(val(r, 'latency_multiplier', 'shieldagent'), 2)}$\times$ & "
            rf"{fmt(val(r, 'latency_multiplier', 'pcg_mas'), 2)}$\times$ \\"
        )

    write_all(
        "appendix_remaining_50_summary.tex",
        longtable(
            label="tab:appendix_remaining_50_summary",
            caption=(
                r"\textbf{Remaining 50-cell summary.} "
                r"Complementary model--dataset cells not shown in the main-text six-cell table."
            ),
            colspec=r"l|ccc|ccc",
            header=[
                r"\multirow{2}{*}{\textbf{Cell}} & "
                r"\multicolumn{3}{c|}{\textbf{Bad accepted claim rate} $\downarrow$} & "
                r"\multicolumn{3}{c}{\textbf{PCG-MAS diagnostics} $\uparrow$} \\",
                r"& No cert. & SHIELD & PCG-MAS & Cert. cov. & Resp.@1 & Safety gain \\",
            ],
            rows=rows_summary,
        ),
    )

    write_all(
        "appendix_remaining_50_r1r4.tex",
        longtable(
            label="tab:appendix_remaining_50_r1r4",
            caption=(
                r"\textbf{Remaining 50-cell R1--R4 diagnostics.} "
                r"Complementary audit, redundancy, responsibility, and control metrics."
            ),
            colspec=r"l|cccc",
            header=[
                r"\textbf{Cell} & Audit cov. $\uparrow$ & Safety gain $\uparrow$ & "
                r"Resp. lift $\uparrow$ & Control gain $\uparrow$ \\",
            ],
            rows=rows_r1r4,
        ),
    )

    write_all(
        "appendix_remaining_50_cost.tex",
        longtable(
            label="tab:appendix_remaining_50_cost",
            caption=(
                r"\textbf{Remaining 50-cell overhead summary.} "
                r"Token and latency multipliers relative to the no-certificate pipeline."
            ),
            colspec=r"l|ccccc",
            header=[
                r"\textbf{Cell} & NoCert tok. & Shield tok. & PCG tok. & Shield lat. & PCG lat. \\",
            ],
            rows=rows_cost,
        ),
    )


# ---------------------------------------------------------------------
# Appendix prompt bank
# ---------------------------------------------------------------------

def prompt_bank() -> None:
    rows = [
        ("Candidate generation", "Produce answer candidates and candidate claims",
         "Given prompt, retrieved context, tool outputs, and task metadata, emit atomic claims, answer draft, cited support identifiers, and uncertainty flags."),
        ("Certificate construction", "Build unified proof-carrying certificate",
         r"Commit evidence hashes, replay pipeline, schema/tool/memory/policy/delegation metadata, and calibrated confidence into \(Z=(c,S,\Pi,\Gamma,p,\mathrm{meta})\)."),
        ("Unified checker", "Verify evidence and execution jointly",
         "Recompute hashes, replay support pipeline, validate schema/tool/policy contract, and run deterministic entailment/checking before accepting a claim."),
        ("SHIELDAGENT baseline", "Policy-only trajectory shielding baseline",
         "Extract action predicates from the trajectory, retrieve relevant safety-policy rules, assign predicate values, verify rule satisfaction, and emit safety label/explanation."),
        ("Redundancy selector", "Enforce independent support paths",
         r"Select \(k\) certificates whose support paths satisfy provenance, tool-overlap, and replayable-overlap separation constraints."),
        ("Audit channel probe", "Estimate finite-sample audit envelope",
         "Probe integrity, replay, checker, and coverage channels to estimate the audit-channel envelope."),
        ("Mask-and-replay", "Estimate responsibility",
         "Mask evidence/tool/schema/memory/policy/delegation components and replay the same trace to estimate acceptance sensitivity."),
        ("Risk controller", "Choose runtime action",
         r"Use certificate-derived risk, utility, token/latency/tool costs, and harm penalty to choose among \texttt{Answer}, \texttt{Verify}, \texttt{Escalate}, and \texttt{Refuse}."),
        ("Colab reconciliation", "Merge remote large-model cells",
         r"Normalize remote \texttt{Llama-3.3-70B} and \texttt{DeepSeek-V3} JSON outputs into the same \(7\times8\) manifest schema used by the local matrix runner."),
    ]

    body = []
    last_stage = None
    for stage, purpose, summary in rows:
        stage_cell = tex_escape(stage) if stage != last_stage else ""
        body.append(rf"{stage_cell} & {tex_escape(purpose)} & {summary} \\")
        last_stage = stage

    write_all(
        "appendix_prompt_bank.tex",
        longtable(
            label="tab:appendix_prompt_bank",
            caption=(
                r"\textbf{Prompt and tool-call templates.} "
                r"Templates used by the PCG-MAS benchmark, grouped by pipeline stage."
            ),
            colspec=r"p{0.16\linewidth}p{0.26\linewidth}p{0.50\linewidth}",
            header=[r"\textbf{Stage} & \textbf{Purpose} & \textbf{Template summary} \\"],
            rows=body,
        ),
    )


def main() -> None:
    main6, all56 = load()
    table_main_six(main6)
    table_r1r4(main6)
    table_cost(main6)
    appendix_remaining(all56, main6)
    prompt_bank()


if __name__ == "__main__":
    main()