"""v3.2 artifact emitter: 33 tables, 10 figures, generated macros, export gate."""
from __future__ import annotations

import os
import re
import shutil
from typing import Any, Callable, Dict, List, Optional

from ..canon import sha256_file, sha256_obj
from ..notation import groups
from ..release import METRIC_VERSION, PCG_MAS_RELEASE
from .contract import ManuscriptContract, parse, verify
from .provenance import (MACRO_FORBIDDEN, SYNTHETIC_STAMP, ProvenanceClass,
                         assert_release_admissible, classify_records)

OUT = "artifacts/v3_0"
TABLES = os.path.join(OUT, "tables", "latex")
CSV = os.path.join(OUT, "tables", "csv")
IMAGES = os.path.join(OUT, "figures", "pdf")
PEND = r"\PEND{}"

#: LaTeX text-mode escapes. Underscores in identifiers such as TEST_FIXTURE,
#: H_support and N_acc are math-mode characters and MUST be escaped.
_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(text: str) -> str:
    """Escape text-mode content. Segments already inside \( ... \) are preserved."""
    parts = re.split(r"(\\\(.*?\\\))", str(text), flags=re.S)
    out = []
    for i, seg in enumerate(parts):
        if i % 2 == 1:                      # inline math: leave as authored
            out.append(seg)
        else:
            out.append("".join(_TEX_ESCAPES.get(ch, ch) for ch in seg))
    return "".join(out)


def _fmt(v, nd=3):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, (int, bool)):
        return str(v)
    return tex_escape(v)


def _table_tex(stem: str, label: str, caption: str, header, rows, provenance: str,
               synthetic: bool) -> str:
    cap = tex_escape(caption)
    if synthetic:
        cap = f"\\textbf{{{tex_escape(SYNTHETIC_STAMP)}.}} " + cap
    al = "l" + "c" * (len(header) - 1)
    out = ["% GENERATED FILE -- do not edit by hand.",
           f"% emitter=pcg.v3.emit  metric_version={METRIC_VERSION}  release={PCG_MAS_RELEASE}",
           f"% provenance-class: {provenance}",
           "\\begin{table}[!ht]", "\\centering",
           f"\\caption{{\\footnotesize \\textsc{{{tex_escape(provenance)}}}. {cap}}}",
           f"\\label{{{label}}}", "\\scriptsize",
           f"\\begin{{tabular}}{{{al}}}", "\\toprule",
           " & ".join(tex_escape(h) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(" & ".join(_fmt(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def emit_all_tables(contract: ManuscriptContract, metrics: Dict[str, Any],
                    prov_summary: Dict[str, Any]) -> Dict[str, Any]:
    """One file per \\pcgtable call. All 33 are mandatory."""
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(CSV, exist_ok=True)
    synthetic = prov_summary["requires_stamp"]
    written: List[str] = []
    for stem in contract.tables:
        label = "tab:" + stem.split("_", 2)[2] if stem.count("_") >= 2 else "tab:" + stem
        if stem.endswith("notation"):
            header = ["Symbol", "Meaning"]
            rows = [[f"\\({s}\\)", m] for grp in groups().values() for s, m in grp]
            prov = ProvenanceClass.STATIC.value
            cap = "Notation, generated from the machine-readable symbol registry."
            body = _table_tex(stem, label, cap, header, rows, prov, False)
        else:
            prov = (ProvenanceClass.TEST_FIXTURE.value if synthetic
                    else ProvenanceClass.DERIVED_FINAL.value)
            header, rows, cap = _shape_for(stem, metrics)
            body = _table_tex(stem, label, cap, header, rows, prov, synthetic)
        with open(os.path.join(TABLES, f"{stem}.tex"), "w") as fh:
            fh.write(body)
        written.append(stem)
    return {"written": len(written), "expected": len(contract.tables),
            "synthetic_stamped": synthetic}


def _shape_for(stem: str, m: Dict[str, Any]):
    """Table shapes driven by metric artifacts; unmeasured cells render \\PEND{}."""
    a05 = (m.get("a05") or {}).get("by_system") or {}
    a10 = (m.get("a10") or {}).get("by_system") or {}
    if "cost" in stem:
        return (["System", "lat p50 (ms)", "lat p95 (ms)", "model calls", "cost/accepted"],
                [[s, v.get("latency_p50_ms"), v.get("latency_p95_ms"),
                  v.get("model_calls"), v.get("cost_per_accepted_correct")]
                 for s, v in sorted(a10.items())] or [["--"] * 5],
                "Absolute direct cost; overhead reported separately, never subtracted.")
    if any(k in stem for k in ("summary", "six", "r1_r4", "sota", "ablation", "remaining")):
        return (["System", "N", "N_acc", "Coverage", "H_support", "H_exec", "H_joint"],
                [[s, v.get("N"), v.get("N_acc"), v.get("coverage"), v.get("H_support"),
                  v.get("H_exec"), v.get("H_joint")] for s, v in sorted(a05.items())]
                or [["--"] * 7],
                "Generated from canonical per-example records.")
    return (["Field", "Value"],
            [["metric_version", METRIC_VERSION], ["release", PCG_MAS_RELEASE],
             ["status", "generated"]],
            "Protocol/specification artifact generated from the frozen spec.")


def emit_macros(prov_summary: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """The two narrative hooks. Never a number from fixture/mock/smoke/pilot."""
    os.makedirs(TABLES, exist_ok=True)
    forbidden = bool(set(prov_summary["classes"]) & MACRO_FORBIDDEN)
    if forbidden:
        abstract = ("The pre-registered evaluation reports headline harm, uncertainty, "
                    "sample size, and cost from canonical artifacts once the frozen run "
                    "is complete.")
        headline = ("The generated headline-results sentence is emitted from the "
                    "reconciled canonical artifacts.")
    else:
        abstract = ("PCG-MAS reduces accepted harm by \\PEND{} at matched coverage "
                    "(95\\% CI \\PEND{}, $N=$\\PEND{}).")
        headline = ("At matched coverage the certificate contributes \\PEND{} of the "
                    "reduction, with selectivity accounting for \\PEND{}.")
    body = ("% GENERATED FILE -- do not edit by hand.\n"
            f"% provenance_classes={prov_summary['classes']}\n"
            f"% numbers_permitted={not forbidden}\n"
            f"\\newcommand{{\\PCGAbstractResultSentence}}{{{abstract}}}\n"
            f"\\newcommand{{\\PCGHeadlineResultSentence}}{{{headline}}}\n")
    p = os.path.join(TABLES, "generated_results_macros.tex")
    open(p, "w").write(body)
    return {"path": p, "numbers_permitted": not forbidden,
            "reason": ("non-admissible provenance: development text emitted"
                       if forbidden else "admissible provenance")}


def overleaf_export(contract: ManuscriptContract, records: List[Dict[str, Any]],
                    allow_synthetic: bool = False,
                    out_root: Optional[str] = None) -> Dict[str, Any]:
    """BLOCKING gate. Refuses to build a release export from synthetic records."""
    summary = classify_records(records)
    assert_release_admissible(summary, allow_synthetic)
    synthetic = summary["requires_stamp"]
    if out_root is not None:
        root = out_root
    else:
        root = os.path.join(OUT, "overleaf_export_SYNTHETIC" if synthetic else "overleaf_export")
    sub = os.path.join(root, contract.project_subdir)
    t_dir, i_dir = os.path.join(sub, "tables"), os.path.join(sub, "images")
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(t_dir, exist_ok=True)
    os.makedirs(i_dir, exist_ok=True)
    copied_t = copied_f = 0
    for stem in contract.tables:
        src = os.path.join(TABLES, f"{stem}.tex")
        if os.path.exists(src):
            shutil.copy2(src, t_dir)
            copied_t += 1
    mac = os.path.join(TABLES, "generated_results_macros.tex")
    if os.path.exists(mac):
        shutil.copy2(mac, t_dir)
    for fig in contract.figures:
        src = os.path.join(IMAGES, fig)
        if os.path.exists(src):
            shutil.copy2(src, i_dir)
            copied_f += 1
    manifest = {"release": PCG_MAS_RELEASE, "synthetic": synthetic,
                "provenance": summary, "tables": copied_t, "figures": copied_f,
                "layout": f"{contract.project_subdir}/{{tables,images}}"}
    open(os.path.join(root, "MANIFEST.json"), "w").write(sha256_obj(manifest) + "\n" +
                                                         str(manifest))
    return {**manifest, "root": root}
