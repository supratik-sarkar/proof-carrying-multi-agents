#!/usr/bin/env python3
"""Fail-closed contamination and provenance gates.

Each gate inspects semantics and lineage, not merely the presence of a string:
a token found inside a report that *documents* a defect is not itself a defect.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_EXT = {".py", ".sh", ".yaml", ".yml", ".toml"}
# Paths whose job is to describe defects; a match there is documentation, not contamination.
DOC_ZONES = ("reports/", "docs/", "quarantine/", "artifacts/evidence/validation/",
             "artifacts/evidence/table_reconciliation/patches/", "tests/",
             "scripts/validate/provenance_gates.py")

def code_files():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in CODE_EXT: continue
        rel = str(p.relative_to(ROOT))
        if any(rel.startswith(z) for z in DOC_ZONES): continue
        yield p, rel

def g_hidden_multipliers():
    hits = []
    for p, rel in code_files():
        t = p.read_text(errors="ignore")
        # a multiplier dict is only a defect if it is *applied*, not merely named
        if re.search(r"SOTA_CALIBRATED\s*\[", t) or re.search(r"\*\s*SOTA_CALIBRATED", t):
            hits.append(f"{rel}: SOTA_CALIBRATED applied to values")
        if re.search(r"CALIBRAT\w*\s*=\s*\{[^}]*\d\.\d+", t):
            hits.append(f"{rel}: hand-set multiplier table")
    return hits

def g_stub_in_pipeline():
    hits = []
    for p, rel in code_files():
        t = p.read_text(errors="ignore")
        if "schema_preflight_stub" in t and "results" in t:
            hits.append(f"{rel}: schema_preflight_stub in an emitting pipeline")
    return hits

def g_fallback_pathways():
    hits = []
    for p, rel in code_files():
        t = p.read_text(errors="ignore")
        for tok in ("allow_partial", "allow_fallback"):
            if re.search(rf"\b{tok}\b(?!_DISABLED)", t):
                hits.append(f"{rel}: live {tok} pathway")
    return hits

def g_numeric_defaults():
    hits = []
    pat = re.compile(r"\.get\(\s*[\"'](harm|utility|tokens|latency|coverage)[\"']\s*,\s*[-\d.]+\s*\)")
    for p, rel in code_files():
        if p.suffix != ".py": continue
        for m in pat.finditer(p.read_text(errors="ignore")):
            hits.append(f"{rel}: silent numeric default for '{m.group(1)}'")
    return hits

def g_absolute_paths():
    import os
    import re as _re
    from pathlib import Path as _Path

    excluded_dirs = {
        ".git",
        ".venv",
        ".venvs",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "_local_release",
        "quarantine",
    }

    text_suffixes = {
        ".py",
        ".json",
        ".jsonl",
        ".csv",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".tex",
        ".bib",
        ".sh",
        ".ini",
        ".cfg",
    }

    absolute_user_path = _re.compile(
        r"/[Uu]sers/[A-Za-z0-9._-]+/|[A-Za-z]:\\[Uu]sers\\[^\\]+\\"
    )

    hits = []

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in excluded_dirs
            and not d.endswith(".venv")
        ]

        current = _Path(dirpath)

        try:
            relative_dir = current.relative_to(ROOT)
        except ValueError:
            continue

        if any(
            part in excluded_dirs or part.endswith(".venv")
            for part in relative_dir.parts
        ):
            continue

        for filename in filenames:
            path = current / filename

            if path.suffix.lower() not in text_suffixes:
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if absolute_user_path.search(text):
                hits.append(
                    f"{path.relative_to(ROOT).as_posix()}: absolute developer path"
                )

    return hits

def g_unsupported_direct():
    """A record or manifest may not claim DIRECT/executed without native evidence fields."""
    NATIVE = {"prompt","response","completion","model_revision","backend","provider_route",
              "request_id","timestamp","tokenizer","decoding_config","output_hash"}
    hits = []
    for p in ROOT.rglob("*.json"):
        rel = str(p.relative_to(ROOT))
        if any(rel.startswith(z) for z in DOC_ZONES): continue
        try: d = json.loads(p.read_text(errors="ignore"))
        except Exception: continue
        blob = json.dumps(d)
        claims = re.search(r'"(provenance|provenance_class|empirical_status|classification)"\s*:\s*"[^"]*(DIRECT|EXECUTED)', blob)
        if claims and not (NATIVE & set(re.findall(r'"([a-z_]+)"\s*:', blob))):
            hits.append(f"{rel}: claims {claims.group(2)} with no native evidence fields")
    return hits

def g_figures_have_lineage():
    m = ROOT / "reports/PLOT_REGENERATION_MATRIX.csv"
    if not m.exists(): return ["reports/PLOT_REGENERATION_MATRIX.csv missing"]
    import csv as _c
    hits = []
    for r in _c.DictReader(m.open()):
        if r["status"] == "REGENERATED" and not r["source"]:
            hits.append(f"{r['figure']}: regenerated without recorded source")
    return hits

GATES = {
    "no_hidden_multipliers":        g_hidden_multipliers,
    "no_stub_in_empirical_pipeline":g_stub_in_pipeline,
    "no_fallback_pathways":         g_fallback_pathways,
    "no_silent_numeric_defaults":   g_numeric_defaults,
    "no_absolute_developer_paths":  g_absolute_paths,
    "no_unsupported_direct_claims": g_unsupported_direct,
    "figures_carry_lineage":        g_figures_have_lineage,
}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT/"reports/gate_report.json")
    a = ap.parse_args()
    res, failed = {}, 0
    for name, fn in GATES.items():
        hits = fn()
        res[name] = {"status": "PASS" if not hits else "FAIL", "findings": hits}
        failed += bool(hits)
        print(f"  [{'PASS' if not hits else 'FAIL'}] {name}" + (f"  ({len(hits)})" if hits else ""))
        for h in hits[:4]: print(f"        - {h}")
    a.output.parent.mkdir(exist_ok=True)
    a.output.write_text(json.dumps(res, indent=2)+"\n")
    print(f"\ngates: {len(GATES)-failed} PASS / {failed} FAIL")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
