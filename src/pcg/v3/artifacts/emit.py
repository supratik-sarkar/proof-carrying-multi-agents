"""Emit CSV + LaTeX from metric artifacts. No manuscript cell is hand-entered."""
from __future__ import annotations
import csv, io, os
from typing import Any, Dict, List, Optional, Sequence

from ..canon import sha256_file
from ..release import METRIC_VERSION

CSV_DIR = "artifacts/v3_0/tables/csv"
TEX_DIR = "artifacts/v3_0/tables/latex"
PEND = r"\PEND{}"


def fmt(v: Any, nd: int = 3) -> str:
    """None -> \\PEND{}. Undefined is NEVER rendered as 0."""
    if v is None:
        return PEND
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_csv(stem: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    os.makedirs(CSV_DIR, exist_ok=True)
    p = os.path.join(CSV_DIR, f"{stem}.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if c is None else c for c in r])
    return p


def write_latex(stem: str, header: Sequence[str], rows: Sequence[Sequence[Any]],
                caption: str, label: str, provenance: str,
                align: Optional[str] = None, nd: int = 3) -> str:
    os.makedirs(TEX_DIR, exist_ok=True)
    al = align or ("l" + "c" * (len(header) - 1))
    buf = io.StringIO()
    buf.write("% GENERATED FILE -- do not edit by hand.\n")
    buf.write(f"% source: pcg.v3.artifacts.emit  metric_version={METRIC_VERSION}\n")
    buf.write("\\begin{table}[!ht]\n\\centering\n")
    buf.write(f"\\caption{{\\footnotesize \\textsc{{{provenance}}}. {caption}}}\n")
    buf.write(f"\\label{{{label}}}\n\\scriptsize\n")
    buf.write(f"\\begin{{tabular}}{{{al}}}\n\\toprule\n")
    buf.write(" & ".join(str(h) for h in header) + " \\\\\n\\midrule\n")
    for r in rows:
        buf.write(" & ".join(fmt(c, nd) for c in r) + " \\\\\n")
    buf.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    p = os.path.join(TEX_DIR, f"{stem}.tex")
    open(p, "w").write(buf.getvalue())
    return p


def emit(stem: str, header, rows, caption, label, provenance, nd: int = 3) -> Dict[str, str]:
    c = write_csv(stem, header, rows)
    t = write_latex(stem, header, rows, caption, label, provenance, nd=nd)
    return {"csv": c, "latex": t, "csv_sha256": sha256_file(c), "latex_sha256": sha256_file(t)}
