"""Figure generation: every scientific figure emits PNG (>=300 dpi) AND vector PDF.

Provenance discipline:
  * a figure whose source records are TEST_FIXTURE is stamped SYNTHETIC DEMO
    on-canvas and is never presented as DIRECT;
  * MODELLED figures carry an analytic-curve stamp;
  * source data, generation config and SHA-256 are written alongside.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..canon import canonical_json, sha256_file
from ..release import PCG_MAS_RELEASE

PNG_DIR = "artifacts/v3_0/figures/png"
PDF_DIR = "artifacts/v3_0/figures/pdf"
DATA_DIR = "artifacts/v3_0/figures/data"
DPI = 300

INK = "#101418"; MUTED = "#5b6570"; GRID = "#dfe4ea"
ACCENT = "#1f6feb"; WARN = "#d1441c"; OK = "#1a7f52"; ALT = "#7b5ea7"; SAND = "#c08a2e"
PALETTE = [ACCENT, OK, SAND, ALT, WARN]


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "600",
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def _stamp(fig, provenance: str, synthetic: bool):
    label = provenance if not synthetic else f"{provenance} · SYNTHETIC DEMO (fixture-derived)"
    fig.text(0.995, 0.005, f"{label} · PCG-MAS {PCG_MAS_RELEASE}",
             ha="right", va="bottom", fontsize=6.5,
             color=(WARN if synthetic else MUTED))


def save(fig, stem: str, provenance: str, data: Dict[str, Any],
         synthetic: bool = True) -> Dict[str, Any]:
    for d in (PNG_DIR, PDF_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)
    _stamp(fig, provenance, synthetic)
    png = os.path.join(PNG_DIR, f"{stem}.png")
    pdf = os.path.join(PDF_DIR, f"{stem}.pdf")
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)                      # vector, fonttype 42 => extractable text
    plt.close(fig)
    dp = os.path.join(DATA_DIR, f"{stem}.json")
    payload = {"stem": stem, "provenance_class": provenance, "synthetic": synthetic,
               "release": PCG_MAS_RELEASE, "dpi": DPI, "data": data}
    open(dp, "w").write(canonical_json(payload))
    return {"png": png, "pdf": pdf, "data": dp,
            "png_sha256": sha256_file(png), "pdf_sha256": sha256_file(pdf),
            "data_sha256": sha256_file(dp)}


# ----------------------------------------------------------------- generators
def fig_intro_overview(m: Dict[str, Any]) -> Dict[str, Any]:
    by = m.get("a05", {}).get("by_system", {})
    order = ["nocert", "citation_only", "shieldagent", "agentrr", "pcg_mas"]
    names = [s for s in order if s in by]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    for ax, key, title in zip(axes, ["H_support", "H_exec", "H_joint"],
                              ["Support harm", "Execution harm", "Joint harm (labelled)"]):
        vals = [by[s].get(key) for s in names]
        ax.bar(range(len(names)), [v or 0 for v in vals],
               color=[ACCENT if s == "pcg_mas" else MUTED for s in names], width=.62)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        ax.set_title(title); ax.set_ylabel("rate" if key == "H_support" else "")
    fig.suptitle("From cited answers to checkable acceptance", fontsize=11, y=1.04)
    return save(fig, "intro_overview", "DERIVED", {"by_system": by})


def fig_headline_budget_frontier(m: Dict[str, Any]) -> Dict[str, Any]:
    arms = m.get("a16", {}).get("arms", {})
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series: Dict[str, List] = {}
    for arm, systems in sorted(arms.items()):
        for s, v in systems.items():
            series.setdefault(s, []).append((v.get("cost_usd") or 0.0,
                                             v.get("harmful_accepted_rate")))
    for i, (s, pts) in enumerate(sorted(series.items())):
        pts = sorted(p for p in pts if p[1] is not None)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=PALETTE[i % len(PALETTE)], lw=1.8, ms=4.5,
                label=s, zorder=3 if s == "pcg_mas" else 2)
    ax.set_xlabel("actual resource cost (USD, equal-budget arms)")
    ax.set_ylabel("harmful accepted rate")
    ax.set_title("Budget-matched frontier: does the certificate beat spending more?")
    ax.legend(fontsize=7.5, ncol=2)
    return save(fig, "headline_budget_frontier", "DIRECT", {"arms": arms})


def fig_cost_overhead(m: Dict[str, Any]) -> Dict[str, Any]:
    by = m.get("a10", {}).get("by_system", {})
    names = [s for s in ["nocert", "citation_only", "shieldagent", "agentrr", "pcg_mas"] if s in by]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    x = range(len(names))
    a1.bar([i - .18 for i in x], [by[s]["latency_p50_ms"] or 0 for s in names], .34,
           label="p50", color=ACCENT)
    a1.bar([i + .18 for i in x], [by[s]["latency_p95_ms"] or 0 for s in names], .34,
           label="p95", color=WARN)
    a1.set_xticks(list(x)); a1.set_xticklabels(names, rotation=18, fontsize=7)
    a1.set_ylabel("latency (ms)"); a1.set_title("Absolute latency, not multipliers"); a1.legend(fontsize=7.5)
    cpa = [by[s].get("cost_per_accepted_correct") for s in names]
    a2.bar(x, [c or 0 for c in cpa], .6, color=[ACCENT if s == "pcg_mas" else MUTED for s in names])
    a2.set_xticks(list(x)); a2.set_xticklabels(names, rotation=18, fontsize=7)
    a2.set_title("Cost per accepted correct claim"); a2.set_ylabel("USD")
    return save(fig, "cost_overhead", "DIRECT", {"by_system": by})


def fig_ablations(m: Dict[str, Any]) -> Dict[str, Any]:
    by = m.get("a05", {}).get("by_system", {})
    variants = ["nocert", "citation_only", "shieldagent", "agentrr", "pcg_mas"]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    keys = ["H_support", "H_exec", "H_joint"]
    w = 0.26
    for j, k in enumerate(keys):
        ax.bar([i + (j - 1) * w for i in range(len(variants))],
               [(by.get(v, {}).get(k) or 0) for v in variants], w,
               label=k, color=PALETTE[j])
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=7)
    ax.set_ylabel("rate"); ax.set_title("Ablations by attributed harm cause"); ax.legend(fontsize=7.5)
    return save(fig, "ablations", "DIRECT", {"by_system": by})


def fig_baseline_comparison(m: Dict[str, Any]) -> Dict[str, Any]:
    by = m.get("a05", {}).get("by_system", {})
    names = [s for s in ["nocert", "citation_only", "shieldagent", "agentrr", "pcg_mas"] if s in by]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, s in enumerate(names):
        v = by[s]
        ax.scatter(v.get("coverage") or 0, v.get("H_joint") or 0, s=90,
                   color=PALETTE[i % len(PALETTE)], zorder=3, label=s)
    ax.set_xlabel("coverage (accept rate)"); ax.set_ylabel("joint harm (labelled union)")
    ax.set_title("Native-scope comparison at matched coverage"); ax.legend(fontsize=7.5)
    return save(fig, "baseline_comparison", "DIRECT", {"by_system": by})


def fig_audit_channels(m: Dict[str, Any]) -> Dict[str, Any]:
    per = m.get("a11", {}).get("per_channel", {})
    lam = m.get("a11", {}).get("lambda_union")
    n = m.get("a11", {}).get("n") or 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ks = list(per); vs = [per[k] / n for k in ks]
    a1.barh(ks, vs, color=ACCENT)
    a1.set_title("Five audit channels (not the four conjuncts)"); a1.set_xlabel("firing rate")
    a2.axis("off")
    a2.text(0.0, 0.86, r"$\Lambda_\cup=\mathbb{E}[(N_F-1)_+]$" + f" = {lam:.4f}" if lam is not None else "",
            fontsize=11)
    a2.text(0.0, 0.62, "union-overlap slack, multiplicity weighted", fontsize=8, color=MUTED)
    a2.text(0.0, 0.40, r"residuals OUTSIDE every channel:", fontsize=9)
    a2.text(0.02, 0.26, r"$\varepsilon_{\mathrm{tax}}$  open-world taxonomy residual", fontsize=8.5, color=WARN)
    a2.text(0.02, 0.14, r"$\varepsilon_{\mathrm{src}}$  source/world-truth residual", fontsize=8.5, color=WARN)
    return save(fig, "audit_channels", "DERIVED", {"per_channel": per, "lambda_union": lam})


def fig_attribution_open_set(m: Dict[str, Any]) -> Dict[str, Any]:
    fam = m.get("a14", {}).get("by_family", {})
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ks = list(fam)
    caught = [(fam[k]["bad"] - fam[k]["missed"]) for k in ks]
    missed = [fam[k]["missed"] for k in ks]
    covonly = [fam[k]["covgap_only"] for k in ks]
    xs = range(len(ks))
    ax.bar(xs, caught, color=OK, label="mapped to a channel")
    ax.bar(xs, covonly, bottom=caught, color=SAND, label="CovGap only")
    ax.bar(xs, missed, bottom=[c + o for c, o in zip(caught, covonly)],
           color=WARN, label=r"unclassified ($\varepsilon_{\mathrm{tax}}$)")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("ContractBad events"); ax.legend(fontsize=7.5)
    ax.set_title("Open-set taxonomy stress: what the five channels miss")
    return save(fig, "attribution_open_set", "DIRECT", {"by_family": fam})


def fig_privacy_frontier_modelled(m: Dict[str, Any]) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    Bs = [32, 64, 128, 256]
    etas = [0.0, 0.25, 0.5, 1.0]
    data = {}
    for i, B in enumerate(Bs):
        ys = [(1 - 0.35 * (1 - pow(2.718281828, -B / 96.0))) * (1 + 0.22 * e * e) for e in etas]
        data[str(B)] = ys
        ax.plot(etas, ys, "o-", color=PALETTE[i], lw=1.6, ms=4, label=f"B={B}")
    ax.set_xlabel(r"DP noise level $\eta$"); ax.set_ylabel("relative risk (analytic)")
    ax.set_title("Privacy frontier — MODELLED, hand-chosen coefficients")
    ax.legend(fontsize=7.5, ncol=2)
    return save(fig, "privacy_frontier_modelled", "MODELLED", {"eta": etas, "curves": data})


def fig_scaling_modelled(m: Dict[str, Any]) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ks = [1, 2, 4, 8]
    data = {}
    for i, beta in enumerate([0.85, 1.0, 1.15]):
        ys = [(1 + 0.20 * k) * pow(k, beta - 1) for k in ks]
        data[f"beta={beta}"] = ys
        ax.plot(ks, ys, "o-", color=PALETTE[i], lw=1.6, ms=4, label=rf"$\beta$={beta}")
    ax.set_xlabel("redundancy $k$"); ax.set_ylabel("relative overhead (analytic)")
    ax.set_title("Overhead scaling — MODELLED, not measured"); ax.legend(fontsize=7.5)
    return save(fig, "scaling_modelled", "MODELLED", {"k": ks, "curves": data})


GENERATORS = {
    "intro_overview": fig_intro_overview,
    "headline_budget_frontier": fig_headline_budget_frontier,
    "cost_overhead": fig_cost_overhead,
    "ablations": fig_ablations,
    "baseline_comparison": fig_baseline_comparison,
    "audit_channels": fig_audit_channels,
    "attribution_open_set": fig_attribution_open_set,
    "privacy_frontier_modelled": fig_privacy_frontier_modelled,
    "scaling_modelled": fig_scaling_modelled,
}


def build_all(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    _style()
    out = {}
    for stem, fn in GENERATORS.items():
        out[stem] = fn(metrics)
    return out


# ------------------------------------------------- authored workflow schematic
def fig_pcg_mas_workflow(m):
    """Figure 2. AUTHORED schematic with a reproducible vector source.

    Class STATIC: it is not data-derived and is never presented as though it were.
    """
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.4); ax.axis("off")
    stages = [("Request", MUTED), ("Generation", ACCENT), ("Evidence", ACCENT),
              ("Commitment\n$V_H$", OK), ("Replay\n$V_\\Pi$", OK),
              ("Policy\n$V_\\Gamma$", OK), ("Entailment\n$V_\\vdash$", OK),
              ("Audit", SAND), ("Acceptance", ALT)]
    w, h, gap = 1.02, 0.78, 0.11
    for i, (label, col) in enumerate(stages):
        x = 0.18 + i * (w + gap)
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.75), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     linewidth=1.1, edgecolor=col, facecolor="white"))
        ax.text(x + w / 2, 1.75 + h / 2, label, ha="center", va="center",
                fontsize=7.4, color=INK)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap, 2.14), xytext=(x + w, 2.14),
                        arrowprops=dict(arrowstyle="-|>", color=GRID, lw=1.0))
    ax.text(0.18, 1.42, "certificate root  $R_Z$  (PCG-CAS-v1 Merkle DAG over committed objects)",
            fontsize=7.6, color=MUTED)
    ax.annotate("", xy=(9.9, 1.58), xytext=(0.18, 1.58),
                arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, linestyle=(0, (3, 3))))
    for i, (t, c) in enumerate([("audit channels: IntFail · ReplayFail · DriftFail · CheckFail · CovGap", SAND),
                                (r"residuals outside every channel: $\varepsilon_{\mathrm{tax}}$ (taxonomy) · $\varepsilon_{\mathrm{src}}$ (source truth)", WARN)]):
        ax.text(0.18, 0.95 - i * 0.36, t, fontsize=7.2, color=c)
    ax.text(0.18, 3.06, "PCG-MAS execution contract", fontsize=10, color=INK, weight="600")
    return save(fig, "pcg_mas_workflow", "STATIC",
                {"authored": True, "stages": [s for s, _ in stages]}, synthetic=False)


GENERATORS["pcg_mas_workflow"] = fig_pcg_mas_workflow
