#!/usr/bin/env python3
"""Regenerate figures derivable from the 56-cell records, under the frozen contract.

Every output is classified REGENERATED_FROM_UNKNOWN_PROVENANCE_56_CELL and
NOT_SAFE_FOR_EMPIRICAL_MANUSCRIPT_USE. Figures with no defensible input are
reported BLOCKED; none is fabricated or substituted.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/repro")); sys.path.insert(0, str(ROOT / "scripts/figures"))
from canonical_metrics import load, by_cell, cell_row, SYSTEMS, ABLATIONS, ablation_harm
import cosmetic_contract as CC

RECS = ROOT / "artifacts/evidence/source_records/per_example_records.jsonl"
OUT  = ROOT / "results/figures"
MANU_DIR = "manuscript"
SUPP_DIR = "supplementary"
CLASS = "REGENERATED_FROM_UNKNOWN_PROVENANCE_56_CELL"
UNSAFE = "NOT_SAFE_FOR_EMPIRICAL_MANUSCRIPT_USE"
HEADLINE = ["phi-3.5-mini__FEVER","qwen2.5-7B__HotpotQA","Llama-3.1-8B__PubMedQA",
            "Gemma-2-9b-it__TAT-QA","Llama-3.3-70B__ToolBench","deepseek-v3__WebLINX"]
SHORT = {c: c.split("__")[0] + "\n" + c.split("__")[1] for c in HEADLINE}

# Manuscript figures with no defensible input in this repository.
BLOCKED = {
 "pcg-mas-workflow.pdf": ("MISSING_ENGINEERING",
   "Hand-authored schematic; no data input required. No vector source exists in the repository "
   "(only workflow/*.png). Rule E forbids promoting a raster to the PDF output. Needs a vector source authored."),
 "pcg-mas_r1_to_r4.svg": ("MISSING_SCIENTIFIC_INPUT",
   "Composite R1-R4 panel. R2 requires a redundancy sweep over k; the records contain no k or "
   "redundancy field (verified by full field inventory). R1 drift channel likewise absent."),
 "r1b_five_channel_audit.pdf": ("MISSING_SCIENTIFIC_INPUT",
   "Five-channel audit requires DriftFail. Field inventory confirms no drift field exists in any record. "
   "Drift is observable only by re-calling the live environment, which is prohibited here."),
 "r3b_open_mixed.pdf": ("MISSING_SCIENTIFIC_INPUT",
   "Panel requires open top-2 and multi-label F1. Records provide top1_correct, top3_correct, "
   "open_set_unknown, unresolved and margin only. The derivable subset is published as "
   "responsibility_diagnostics.pdf; the manuscript panel is not fully specified."),
 "r4b_privacy_frontier.pdf": ("MISSING_SCIENTIFIC_INPUT",
   "Eq. (138)-(139) disclose the FORM but not the coefficients (alpha, beta, B0, tau, sigma, epsilon0, "
   "U0, u_B, u_eta, u_R). scripts/figures/make_r4_privacy_frontier.py implements a DIFFERENT model "
   "(rho_hat = 1 + 1.5*overlap, Gaussian noise) and needs a --cert-summary-jsonl input that does not exist. "
   "Choosing coefficients would be inventing values."),
 "r5b_scaling.pdf": ("MISSING_SCIENTIFIC_INPUT",
   "Eq. (140)-(141) disclose the FORM but not T0, a, b, c, theta, zeta. "
   "scripts/figures/make_r5_scaling.py implements a different linear cost model whose own comment reads "
   "'Replace constants with measured values from actual timed checker calls'. Values unavailable."),
 "intro_hero_v5.pdf": ("MISSING_ENGINEERING",
   "No v5 specification exists anywhere in the repository (searched configs/, artifacts/, scripts/). "
   "A v4 generator exists (src/pcg/eval/intro_hero_v4.py) but whether v5 differs only by naming/layout "
   "cannot be established, so Rule A cannot be applied without guessing at content."),
 "appendix_hero_v5.pdf": ("MISSING_SCIENTIFIC_INPUT",
   "Requires six baselines. PRISM/ATLAS, PCN-Rec and CLBC operate on other task families "
   "(model-driven engineering, recommendation, covert-channel bounds) and cannot produce this metric. "
   "No v5 specification exists either."),
}

MANUSCRIPT_STEMS = {"ablations"}          # the one manuscript figure this repair regenerates

def _save(fig, stem, rows, out):
    sub = MANU_DIR if stem in MANUSCRIPT_STEMS else SUPP_DIR
    d = out / sub; d.mkdir(parents=True, exist_ok=True)
    geo = CC.MANUSCRIPT_GEOMETRY.get(stem)
    bbox = geo["bbox_inches"] if geo else CC.SAVE["bbox_inches"]
    paths = []
    for ext in CC.SAVE["formats"]:
        p = d / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches=bbox, dpi=CC.SAVE["dpi"])
        paths.append(p)
    plt.close(fig)
    w, h = fig.get_size_inches()
    rows.append({"figure": stem, "class": "manuscript" if stem in MANUSCRIPT_STEMS else "supplementary",
                 "outputs": ";".join(str(p.relative_to(out)) for p in paths),
                 "generator": "scripts/repro/regenerate_figures.py",
                 "source": "artifacts/evidence/source_records/per_example_records.jsonl",
                 "classification": CLASS, "safety": UNSAFE, "status": "REGENERATED",
                 "width_in": round(w,3), "height_in": round(h,3), "dpi": CC.SAVE["dpi"],
                 "cosmetic_contract": "APPLIED"})
    return paths

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=RECS)
    ap.add_argument("--outdir",  type=Path, default=OUT)
    a = ap.parse_args()
    if not a.records.exists(): raise SystemExit(f"BLOCKED: records absent at {a.records}")
    recs = load(a.records); g = by_cell(recs)
    a.outdir.mkdir(parents=True, exist_ok=True)
    for stale in a.outdir.rglob("*"):
        if stale.is_file(): stale.unlink()
    CC.apply(matplotlib); CC.apply_manuscript_fonts(matplotlib); rows = []

    # 1 — headline harm by system, six cells, clean vs adversarial
    for cond in ("clean", "adversarial"):
        fig, ax = plt.subplots(figsize=CC.FIGSIZE["panel_row"])
        n, w = len(SYSTEMS), 0.13
        for i, s in enumerate(SYSTEMS):
            vals = [cell_row(g[c], cond)[f"harm::{s}"] for c in HEADLINE]
            xs = [j + (i - n/2)*w for j in range(len(HEADLINE))]
            ax.bar(xs, vals, width=w, label=s, color=CC.SYSTEM_COLORS[s],
                   edgecolor=CC.AXES["edgecolor"], linewidth=0.6)
        ax.set_xticks(range(len(HEADLINE))); ax.set_xticklabels([SHORT[c] for c in HEADLINE])
        ax.set_ylabel("accepted harm rate"); ax.set_title(f"Accepted harm by system — {cond}")
        ax.grid(axis="y", color=CC.AXES["grid_color"], alpha=CC.AXES["grid_alpha"], lw=CC.AXES["grid_lw"])
        ax.set_axisbelow(True); ax.legend(ncol=6, frameon=False)
        for sp in ax.spines.values(): sp.set_linewidth(CC.SPINE_LW)
        _save(fig, f"headline_harm_{cond}", rows, a.outdir)

    # 2 — safety/cost trade-off
    fig, ax = plt.subplots(figsize=CC.FIGSIZE["compact"])
    for s in ("NoCert","ShieldAgent","PCG-MAS"):
        xs = [cell_row(g[c],"clean")[f"tokens_x::{s}"] if s!="NoCert" else 1.0 for c in HEADLINE]
        ys = [cell_row(g[c],"clean")[f"harm::{s}"] for c in HEADLINE]
        ax.scatter(xs, ys, s=70, label=s, color=CC.SYSTEM_COLORS[s],
                   edgecolor=CC.AXES["edgecolor"], linewidth=0.6, zorder=3)
    ax.set_xlabel("token multiplier vs NoCert"); ax.set_ylabel("accepted harm rate")
    ax.set_title("Safety–cost trade-off (six headline cells)")
    ax.grid(color=CC.AXES["grid_color"], alpha=CC.AXES["grid_alpha"], lw=CC.AXES["grid_lw"])
    ax.set_axisbelow(True); ax.legend(frameon=False)
    for sp in ax.spines.values(): sp.set_linewidth(CC.SPINE_LW)
    _save(fig, "safety_cost_tradeoff", rows, a.outdir)

    # 3 — ablations
    fig, ax = plt.subplots(figsize=CC.MANUSCRIPT_GEOMETRY["ablations"]["canvas_in"])
    cl = [r for r in recs if r["condition"]=="clean"]; ad = [r for r in recs if r["condition"]=="adversarial"]
    xs = range(len(ABLATIONS))
    ax.bar([x-0.2 for x in xs], [ablation_harm(cl,a_) for a_ in ABLATIONS], width=0.4,
           label="clean", color=CC.SYSTEM_COLORS["PCG-MAS"], edgecolor=CC.AXES["edgecolor"], linewidth=0.6)
    ax.bar([x+0.2 for x in xs], [ablation_harm(ad,a_) for a_ in ABLATIONS], width=0.4,
           label="adversarial", color=CC.SYSTEM_COLORS["NoCert"], edgecolor=CC.AXES["edgecolor"], linewidth=0.6)
    ax.set_xticks(list(xs)); ax.set_xticklabels(ABLATIONS, rotation=30, ha="right")
    ax.set_ylabel("accepted harm rate"); ax.set_title("Ablations")
    ax.grid(axis="y", color=CC.AXES["grid_color"], alpha=CC.AXES["grid_alpha"], lw=CC.AXES["grid_lw"])
    ax.set_axisbelow(True); ax.legend(frameon=False)
    for sp in ax.spines.values(): sp.set_linewidth(CC.SPINE_LW)
    _save(fig, "ablations", rows, a.outdir)

    # 4 — coverage & responsibility
    fig, ax = plt.subplots(figsize=CC.FIGSIZE["compact"])
    cov = [cell_row(g[c],"clean")["coverage"] for c in HEADLINE]
    rsp = [cell_row(g[c],"clean")["responsibility@1"] for c in HEADLINE]
    xs = range(len(HEADLINE))
    ax.bar([x-0.2 for x in xs], cov, width=0.4, label="audit coverage",
           color=CC.CHANNEL_COLORS["coverage"], edgecolor=CC.AXES["edgecolor"], linewidth=0.6)
    ax.bar([x+0.2 for x in xs], rsp, width=0.4, label="responsibility@1",
           color=CC.CHANNEL_COLORS["replay"], edgecolor=CC.AXES["edgecolor"], linewidth=0.6)
    ax.set_xticks(list(xs)); ax.set_xticklabels([SHORT[c] for c in HEADLINE])
    ax.set_ylabel("rate"); ax.set_title("Audit coverage and responsibility@1")
    ax.grid(axis="y", color=CC.AXES["grid_color"], alpha=CC.AXES["grid_alpha"], lw=CC.AXES["grid_lw"])
    ax.set_axisbelow(True); ax.legend(frameon=False)
    for sp in ax.spines.values(): sp.set_linewidth(CC.SPINE_LW)
    _save(fig, "coverage_responsibility", rows, a.outdir)

    # 5 — checker calibration (from checker.entailment_score / entailment_true)
    fig, ax = plt.subplots(figsize=CC.FIGSIZE["compact"])
    bins = [i/10 for i in range(11)]
    for cond, col in (("clean", CC.SYSTEM_COLORS["PCG-MAS"]), ("adversarial", CC.SYSTEM_COLORS["NoCert"])):
        rs = [r for r in recs if r["condition"] == cond]
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = [r for r in rs if lo <= r["checker"]["entailment_score"] < hi]
            if b:
                xs.append((lo+hi)/2)
                ys.append(sum(1 for r in b if r["checker"]["entailment_true"])/len(b))
        ax.plot(xs, ys, marker="o", lw=1.8, color=col, label=cond)
    ax.plot([0,1],[0,1], ls=":", lw=0.8, color=CC.AXES["grid_color"], label="perfect calibration")
    ax.set_xlabel("entailment score"); ax.set_ylabel("empirical entailment rate")
    ax.set_title("Entailment-checker calibration")
    ax.grid(color=CC.AXES["grid_color"], alpha=CC.AXES["grid_alpha"], lw=CC.AXES["grid_lw"])
    ax.set_axisbelow(True); ax.legend(frameon=False)
    for sp in ax.spines.values(): sp.set_linewidth(CC.SPINE_LW)
    _save(fig, "checker_calibration", rows, a.outdir)

    # 6 — responsibility diagnostics (top1/top3/unresolved/unknown; NOT the manuscript r3b panel,
    #     which additionally requires multi-label F1 and top-2 — absent from the records)
    fig, ax = plt.subplots(figsize=CC.FIGSIZE["panel_row"])
    metrics = [("top1_correct","top-1"),("top3_correct","top-3"),
               ("unresolved","unresolved"),("open_set_unknown","open-set unknown")]
    w2 = 0.2
    for i,(key,lab) in enumerate(metrics):
        vals=[]
        for c in HEADLINE:
            rs=[r for r in g[c] if r["condition"]=="clean"]
            vals.append(sum(1 for r in rs if r["responsibility"][key])/len(rs) if rs else None)
        xs=[j+(i-len(metrics)/2)*w2 for j in range(len(HEADLINE))]
        ax.bar(xs, vals, width=w2, label=lab,
               color=list(CC.CHANNEL_COLORS.values())[i],
               edgecolor=CC.AXES["edgecolor"], linewidth=0.6)
    ax.set_xticks(range(len(HEADLINE))); ax.set_xticklabels([SHORT[c] for c in HEADLINE])
    ax.set_ylabel("rate"); ax.set_title("Responsibility diagnostics (derivable subset)")
    ax.grid(axis="y", color=CC.AXES["grid_color"], alpha=CC.AXES["grid_alpha"], lw=CC.AXES["grid_lw"])
    ax.set_axisbelow(True); ax.legend(ncol=4, frameon=False)
    for sp in ax.spines.values(): sp.set_linewidth(CC.SPINE_LW)
    _save(fig, "responsibility_diagnostics", rows, a.outdir)

    for name, why in BLOCKED.items():
        kind, why = name_kind_reason = BLOCKED[name]
        rows.append({"figure": name, "class": "manuscript", "outputs": "", "generator": "",
                     "source": "", "classification": f"BLOCKED_BY_{kind}",
                     "safety": UNSAFE, "status": "BLOCKED", "width_in": "", "height_in": "",
                     "dpi": "", "cosmetic_contract": "N/A", "reason": why})

    cols = ["figure","class","outputs","generator","source","classification","safety","status",
            "width_in","height_in","dpi","cosmetic_contract","reason"]
    mp = ROOT/"reports/PLOT_REGENERATION_MATRIX.csv"; mp.parent.mkdir(exist_ok=True)
    with mp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (ROOT/"reports/PLOT_COSMETIC_FINGERPRINT.json").write_text(json.dumps(CC.fingerprint(), indent=2)+"\n")

    # machine-readable guard: supplementary output must never be treated as a manuscript asset
    manifest = {
        "schema": "pcg.figure_classification/1",
        "rule": "Figures under supplementary/ are NOT manuscript assets and must never be wired "
                "into the manuscript.",
        "manuscript_dir": f"{a.outdir.relative_to(ROOT)}/{MANU_DIR}",
        "supplementary_dir": f"{a.outdir.relative_to(ROOT)}/{SUPP_DIR}",
        "repository_inferred_manuscript_figures": 10,
        "full_manuscript_figure_universe_validation": "BLOCKED_MAIN_TEX_ABSENT",
        "figures": [
            {"file": f"{sub}/{f.name}", "class": cls,
             "input_classification": "DERIVED_FROM_UNKNOWN_PROVENANCE_56_CELL",
             "safe_for_empirical_manuscript_use": False}
            for sub, cls in ((MANU_DIR, "MANUSCRIPT"), (SUPP_DIR, "SUPPLEMENTARY"))
            for f in sorted((a.outdir / sub).glob("*.pdf"))
        ],
    }
    (a.outdir / "FIGURE_CLASSIFICATION.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ok = sum(1 for r in rows if r["status"]=="REGENERATED")
    print(f"regenerated {ok} figures -> {a.outdir.relative_to(ROOT)}")
    print(f"blocked     {len(BLOCKED)} manuscript figures (reasons in PLOT_REGENERATION_MATRIX.csv)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
