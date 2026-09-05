"""Manuscript artifact registry access + integrity checks."""
from __future__ import annotations
import json, os
from typing import Dict, List, Optional

REGISTRY_PATH = os.environ.get("PCG_REGISTRY", "manuscript_artifact_registry.json")


def load(path: Optional[str] = None) -> dict:
    with open(path or REGISTRY_PATH) as fh:
        return json.load(fh)


def tables(reg: Optional[dict] = None) -> List[dict]:
    return (reg or load())["tables"]


def figures(reg: Optional[dict] = None) -> List[dict]:
    return (reg or load())["figures"]


def check_registry(reg: Optional[dict] = None) -> Dict[str, object]:
    r = reg or load()
    t, f = r["tables"], r["figures"]
    gen_t = [x for x in t if x.get("generator")]
    gen_f = [x for x in f if x.get("generator")]
    return {
        "n_tables": len(t), "n_figures": len(f),
        "tables_expected": 33, "figures_expected": 10,
        "tables_complete": len(t) == 33, "figures_complete": len(f) == 10,
        "tables_with_generator": len(gen_t), "figures_with_generator": len(gen_f),
        "tables_unmapped": [x["label"] for x in t
                            if not x["source_experiments"] and x["provenance_class"] != "STATIC"],
        "figures_unmapped": [x["label"] for x in f
                             if not x["source_experiments"] and x["provenance_class"] != "STATIC"],
        "classes_valid": all(x["provenance_class"] in
                             ("DIRECT", "DERIVED", "MODELLED", "PROTOCOL", "STATIC")
                             for x in t + f),
        "dual_output_declared": all(x.get("png") and x.get("pdf") for x in gen_f),
    }


def check_against_tex(tex_path: str, reg: Optional[dict] = None) -> Dict[str, object]:
    """Labels and figure paths in the registry must match the manuscript."""
    import re
    s = open(tex_path).read()
    tex_tabs = set(re.findall(r"\\label\{(tab:[^}]+)\}", s))
    tex_figs = set(re.findall(r"\\label\{(fig:[^}]+)\}", s))
    imgs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s))
    r = reg or load()
    reg_tabs = {x["label"] for x in r["tables"]}
    for x in r["tables"]:
        reg_tabs |= set(x.get("subtables") or [])
    reg_figs = {x["label"] for x in r["figures"]}
    reg_imgs = {x["manuscript_path"] for x in r["figures"] if x.get("manuscript_path")}
    return {
        "table_labels_match": sorted(reg_tabs ^ tex_tabs) == [],
        "figure_labels_match": sorted(reg_figs ^ tex_figs) == [],
        "figure_paths_present_in_tex": sorted(reg_imgs - imgs) == [],
        "missing_in_registry": sorted(tex_tabs - reg_tabs) + sorted(tex_figs - reg_figs),
        "extra_in_registry": sorted(reg_tabs - tex_tabs) + sorted(reg_figs - tex_figs),
    }
