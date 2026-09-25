"""Parse the v3.2 manuscript for its artifact contract. Never a second typed list.

`\\pcgtable{X}` expands to `\\input{\\tbldir X.tex}` with NO existence guard, so a
missing table file is a hard compile error: all 33 are mandatory. The generated
macro file IS guarded and degrades to non-empirical development text.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_TEX = os.environ.get("PCG_V32_TEX", os.path.join("manuscript",
                                                         "pcg_mas_manuscript_v3-2.tex"))


@dataclass
class ManuscriptContract:
    tex_path: str
    tables: List[str] = field(default_factory=list)     # table_01_... stems
    figures: List[str] = field(default_factory=list)    # foo.pdf basenames
    macros_file: str = "generated_results_macros.tex"
    macros_guarded: bool = False
    tables_guarded: bool = False
    generated_macros: List[str] = field(default_factory=list)
    tbldir: str = "tables/"
    figdir: str = "images/"
    project_subdir: str = "manuscript"

    def to_dict(self) -> dict:
        return vars(self)


def parse(tex_path: str = DEFAULT_TEX) -> ManuscriptContract:
    s = open(tex_path).read()
    tables = re.findall(r"\\pcgtable\{([^}]*)\}", s)
    figures = re.findall(r"\\pcgfigure\{([^}]*)\}", s)
    tbl_def = re.search(r"\\newcommand\{\\pcgtable\}\[1\]\{([^\n]*)\}", s)
    guarded_tables = bool(tbl_def and "IfFileExists" in tbl_def.group(1))
    guarded_macros = bool(re.search(
        r"\\IfFileExists\{\\tbldir\s*generated_results_macros\.tex\}", s))
    macros = sorted(set(re.findall(r"\\(PCG[A-Za-z]+)\s*\{\}", s)))
    sub = "manuscript"
    m = re.search(r"\\IfFileExists\{([A-Za-z0-9_\-]+)/tables/", s)
    if m:
        sub = m.group(1)
    return ManuscriptContract(
        tex_path=tex_path, tables=list(dict.fromkeys(tables)),
        figures=list(dict.fromkeys(figures)), macros_guarded=guarded_macros,
        tables_guarded=guarded_tables, generated_macros=macros, project_subdir=sub)


def verify(contract: ManuscriptContract, tables_dir: str, images_dir: str) -> Dict[str, object]:
    missing_t = [t for t in contract.tables
                 if not os.path.exists(os.path.join(tables_dir, f"{t}.tex"))]
    missing_f = [f for f in contract.figures
                 if not os.path.exists(os.path.join(images_dir, f))]
    macros_ok = os.path.exists(os.path.join(tables_dir, contract.macros_file))
    return {
        "tables_expected": len(contract.tables), "tables_present": len(contract.tables) - len(missing_t),
        "figures_expected": len(contract.figures), "figures_present": len(contract.figures) - len(missing_f),
        "missing_tables": missing_t, "missing_figures": missing_f,
        "macros_present": macros_ok,
        "tables_are_mandatory": not contract.tables_guarded,
        "complete": not missing_t and not missing_f and macros_ok,
    }
