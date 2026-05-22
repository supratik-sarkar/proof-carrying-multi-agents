from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.paper_metric_validation import validate_headline_rows, cells_from_rows


TABLE_FILES = [
    "table_main_six_summary.tex",
    "table_r1_r4_combined.tex",
    "table_cost_overhead_main.tex",
    "table_ablations.tex",
    "table_replay_drift_covgap.tex",
    "table_r3_open_mixed.tex",
    "table_r4_privacy.tex",
    "table_r5_scaling.tex",
    "table_appendix_remaining_50_summary.tex",
    "table_appendix_remaining_50_r1r4.tex",
    "table_appendix_remaining_50_cost.tex",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing rows file: {path}")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    try:
        y = float(x)
        return math.isnan(y) or math.isinf(y)
    except Exception:
        return False


def fmt(x: Any, nd: int = 3) -> str:
    if is_missing(x):
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def fmt_gain(x: Any) -> str:
    if is_missing(x):
        return "NA"
    try:
        return f"{float(x):.2f}$\\times$"
    except Exception:
        return "NA"


def tex_escape(x: Any) -> str:
    s = "unspecified" if is_missing(x) else str(x)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def cell_tex(row: Dict[str, Any]) -> str:
    model = row.get("model")
    dataset = row.get("dataset")
    if model in {None, "", "unknown"}:
        model = "unspecified-model"
    if dataset in {None, "", "unknown"}:
        dataset = "unspecified-dataset"
    return "\\texttt{" + tex_escape(model) + "} / \\texttt{" + tex_escape(dataset) + "}"


def val(row: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in row and not is_missing(row.get(k)):
            return row.get(k)
    return None


def ratio(num: Any, den: Any) -> Any:
    if is_missing(num) or is_missing(den):
        return None
    try:
        d = float(den)
        if abs(d) < 1e-12:
            return None
        return float(num) / d
    except Exception:
        return None


def table(lines: List[str], caption: str, label: str, colspec: str, header: str, rows: Iterable[str]) -> str:
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    out.extend(rows)
    out.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(out)


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def make_main_six(rows: List[Dict[str, Any]], outdir: Path) -> None:
    body = []
    for r in rows:
        body.append(
            f"{cell_tex(r)} & "
            f"{fmt(val(r, 'clean_harm_nocert', 'harm_clean_no_cert'))} & "
            f"{fmt(val(r, 'clean_harm_shieldagent', 'harm_clean_shield'))} & "
            f"{fmt(val(r, 'clean_harm_pcg_mas', 'harm_clean_pcg', 'lhs_accept_and_wrong'))} & "
            f"{fmt(val(r, 'adv_harm_nocert', 'harm_adv_no_cert'))} & "
            f"{fmt(val(r, 'adv_harm_shieldagent', 'harm_adv_shield'))} & "
            f"{fmt(val(r, 'adv_harm_pcg_mas', 'harm_adv_pcg'))} & "
            f"{fmt(val(r, 'responsibility_top1', 'resp_top1'))} & "
            f"{fmt(val(r, 'utility'))} \\\\"
        )
    write(
        outdir / "table_main_six_summary.tex",
        table(
            body,
            "Main six-cell headline summary. Missing measurements are shown as NA.",
            "table_main_six_summary",
            "lrrrrrrrr",
            "Cell & Clean NoCert & Clean ShieldAgent & Clean PCG & Adv. NoCert & Adv. ShieldAgent & Adv. PCG & Resp.@1 & Utility",
            body,
        ),
    )


def make_r1_r4(rows: List[Dict[str, Any]], outdir: Path) -> None:
    body = []
    for r in rows:
        clean_pcg = val(r, "clean_harm_pcg_mas", "harm_clean_pcg", "lhs_accept_and_wrong")
        adv_pcg = val(r, "adv_harm_pcg_mas", "harm_adv_pcg")
        gain_clean = ratio(val(r, "clean_harm_nocert", "harm_clean_no_cert"), clean_pcg)
        gain_adv = ratio(val(r, "adv_harm_nocert", "harm_adv_no_cert"), adv_pcg)
        body.append(
            f"{cell_tex(r)} & {fmt(clean_pcg)} & {fmt(adv_pcg)} & "
            f"{fmt_gain(gain_clean)} & {fmt_gain(gain_adv)} & NA \\\\"
        )
    write(
        outdir / "table_r1_r4_combined.tex",
        table(
            body,
            "R1/R4 combined risk-control summary. Missing measurements are shown as NA.",
            "table_r1_r4_combined",
            "lrrrrr",
            "Cell & Clean harm & Adv. harm & Gain clean & Gain adv. & 95\\% CI",
            body,
        ),
    )


def make_cost(rows: List[Dict[str, Any]], outdir: Path) -> None:
    body = []
    for r in rows:
        body.append(
            f"{cell_tex(r)} & "
            f"{fmt_gain(val(r, 'tokens_nocert'))} & "
            f"{fmt_gain(val(r, 'tokens_shieldagent'))} & "
            f"{fmt_gain(val(r, 'tokens_pcg_mas'))} & "
            f"{fmt_gain(val(r, 'latency_shieldagent'))} & "
            f"{fmt_gain(val(r, 'latency_pcg_mas'))} \\\\"
        )
    write(
        outdir / "table_cost_overhead_main.tex",
        table(
            body,
            "Cost and overhead summary. Missing measurements are shown as NA.",
            "table_cost_overhead_main",
            "lrrrrr",
            "Cell & NoCert tok. & ShieldAgent tok. & PCG tok. & ShieldAgent lat. & PCG lat.",
            body,
        ),
    )


def make_ablations(rows: List[Dict[str, Any]], outdir: Path) -> None:
    body = []
    for r in rows:
        body.append(
            f"{cell_tex(r)} & "
            f"{fmt(val(r, 'p_int_fail'))} & "
            f"{fmt(val(r, 'p_replay_fail'))} & "
            f"{fmt(val(r, 'p_check_fail'))} & "
            f"{fmt(val(r, 'rhs_union'))} & "
            f"{fmt(val(r, 'utility'))} \\\\"
        )
    write(
        outdir / "table_ablations.tex",
        table(
            body,
            "Ablation-style diagnostic summary. Missing measurements are shown as NA.",
            "table_ablations",
            "lrrrrr",
            "Cell & Integrity fail & Replay fail & Check fail & RHS union & Utility",
            body,
        ),
    )


def make_replay_drift(rows: List[Dict[str, Any]], outdir: Path) -> None:
    body = []
    for r in rows:
        body.append(
            f"{cell_tex(r)} & "
            f"{fmt(val(r, 'p_replay_fail'))} & "
            f"{fmt(val(r, 'drift_clean'))} & "
            f"{fmt(val(r, 'covgap_clean'))} & "
            f"{fmt(val(r, 'replay_fresh'))} & "
            f"{fmt(val(r, 'drift_fresh'))} & "
            f"{fmt(val(r, 'covgap_fresh'))} \\\\"
        )
    write(
        outdir / "table_replay_drift_covgap.tex",
        table(
            body,
            "Replay drift and coverage-gap diagnostics. Missing measurements are shown as NA.",
            "table_replay_drift_covgap",
            "lrrrrrr",
            "Cell & Replay clean & Drift clean & CovGap clean & Replay fresh & Drift fresh & CovGap fresh",
            body,
        ),
    )


def make_r3(rows: List[Dict[str, Any]], outdir: Path) -> None:
    body = []
    for r in rows:
        body.append(
            f"{cell_tex(r)} & "
            f"{fmt(val(r, 'responsibility_top1', 'resp_top1'))} & "
            f"{fmt(val(r, 'open_top2'))} & "
            f"{fmt(val(r, 'multilabel_f1'))} & "
            f"{fmt(val(r, 'unknown_acc'))} \\\\"
        )
    write(
        outdir / "table_r3_open_mixed.tex",
        table(
            body,
            "R3 responsibility diagnostics. Missing measurements are shown as NA.",
            "table_r3_open_mixed",
            "lrrrr",
            "Cell & Closed top-1 & Open top-2 & Multi-label F1 & Unknown acc.",
            body,
        ),
    )


def make_na_grid(outdir: Path, filename: str, caption: str, label: str, colspec: str, header: str, row_prefixes: List[str]) -> None:
    ncols = header.count("&") + 1
    body = []
    for left in row_prefixes:
        body.append(left + " & " + " & ".join(["NA"] * (ncols - 1)) + r" \\")
    write(outdir / filename, table(body, caption, label, colspec, header, body))


def make_all_tables(rows: List[Dict[str, Any]], outdir: Path, allow_partial: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    make_main_six(rows, outdir)
    make_r1_r4(rows, outdir)
    make_cost(rows, outdir)
    make_ablations(rows, outdir)
    make_replay_drift(rows, outdir)
    make_r3(rows, outdir)

    if allow_partial:
        make_na_grid(
            outdir,
            "table_r4_privacy.tex",
            "R4 privacy frontier. Partial smoke-test builds show unavailable measurements as NA.",
            "table_r4_privacy",
            "rrrrr",
            r"\(B_{\mathrm{info}}\) & \(\eta\) & \(\widehat\rho\) & Harm & Utility",
            ["32", "64", "128", "256"],
        )
        make_na_grid(
            outdir,
            "table_r5_scaling.tex",
            "R5 scaling summary. Partial smoke-test builds show unavailable measurements as NA.",
            "table_r5_scaling",
            "llll",
            "Variable & Sweep & Token slope & Latency slope",
            [r"Redundancy \(k\)", r"Support size \(|S_0|\)", r"Chain depth \(d\)"],
        )
        make_na_grid(
            outdir,
            "table_appendix_remaining_50_summary.tex",
            "Remaining-cell summary. Partial smoke-test builds show unavailable measurements as NA.",
            "table_appendix_remaining_50_summary",
            "lrrrrrrrr",
            "Cell & Clean NoCert & Clean ShieldAgent & Clean PCG & Adv. NoCert & Adv. ShieldAgent & Adv. PCG & Resp.@1 & Utility",
            [cell_tex(r) for r in rows],
        )
    else:
        make_na_grid(
            outdir,
            "table_r4_privacy.tex",
            "R4 privacy frontier.",
            "table_r4_privacy",
            "rrrrr",
            r"\(B_{\mathrm{info}}\) & \(\eta\) & \(\widehat\rho\) & Harm & Utility",
            ["32", "64", "128", "256"],
        )
        make_na_grid(
            outdir,
            "table_r5_scaling.tex",
            "R5 scaling summary.",
            "table_r5_scaling",
            "llll",
            "Variable & Sweep & Token slope & Latency slope",
            [r"Redundancy \(k\)", r"Support size \(|S_0|\)", r"Chain depth \(d\)"],
        )
        make_na_grid(
            outdir,
            "table_appendix_remaining_50_summary.tex",
            "Remaining-cell summary.",
            "table_appendix_remaining_50_summary",
            "lrrrrrrrr",
            "Cell & Clean NoCert & Clean ShieldAgent & Clean PCG & Adv. NoCert & Adv. ShieldAgent & Adv. PCG & Resp.@1 & Utility",
            [cell_tex(r) for r in rows],
        )

    make_na_grid(
        outdir,
        "table_appendix_remaining_50_r1r4.tex",
        "Remaining-cell R1/R4 appendix summary. Missing measurements are shown as NA.",
        "table_appendix_remaining_50_r1r4",
        "lrrrrr",
        "Cell & Clean harm & Adv. harm & Gain clean & Gain adv. & 95\\% CI",
        [cell_tex(r) for r in rows],
    )
    make_na_grid(
        outdir,
        "table_appendix_remaining_50_cost.tex",
        "Remaining-cell cost appendix summary. Missing measurements are shown as NA.",
        "table_appendix_remaining_50_cost",
        "lrrrrr",
        "Cell & NoCert tok. & ShieldAgent tok. & PCG tok. & ShieldAgent lat. & PCG lat.",
        [cell_tex(r) for r in rows],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PCG-MAS paper LaTeX tables.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    validate_headline_rows(rows, source=str(args.rows), allow_partial=args.allow_partial)

    # Keep compatibility with older code expectations: measured cells are read from rows.
    _ = cells_from_rows(rows)

    make_all_tables(rows, args.outdir, allow_partial=args.allow_partial)

    produced = sorted(args.outdir.glob("*.tex"))
    print(f"Wrote {len(produced)} LaTeX tables to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
