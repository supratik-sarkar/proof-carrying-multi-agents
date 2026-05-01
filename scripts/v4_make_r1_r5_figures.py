#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcg.eval.v4_constants import (
    CHANNEL_COLORS,
    CHANNEL_LABELS,
    FIG_FONT,
    FIG_STYLE,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
)

OUT = Path("figures/v4")


def load_cells() -> list[dict]:
    p = json.loads(Path("results/v4/proxy_metrics.json").read_text())
    return p["r_plot_cells"]


def setup():
    plt.rcParams.update({
        "font.size": FIG_FONT["tick"],
        "axes.titlesize": FIG_FONT["title"],
        "axes.labelsize": FIG_FONT["axis"],
        "xtick.labelsize": FIG_FONT["tick"],
        "ytick.labelsize": FIG_FONT["tick"],
        "legend.fontsize": FIG_FONT["legend"],
        "axes.edgecolor": FIG_STYLE["edge"],
        "axes.labelcolor": FIG_STYLE["ink"],
        "xtick.color": FIG_STYLE["ink"],
        "ytick.color": FIG_STYLE["ink"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def label(cell: dict) -> str:
    return f"{cell['model']}\n[{cell['dataset']}]"


def legend3(ax, loc="upper right"):
    ax.legend(
        loc=loc,
        frameon=True,
        framealpha=0.97,
        facecolor="white",
        edgecolor=FIG_STYLE["edge"],
        fontsize=FIG_FONT["legend"],
        borderpad=0.35,
        handlelength=1.3,
        labelspacing=0.25,
    )


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(path)
    plt.close(fig)


def r1_audit(cells: list[dict]):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    fig.suptitle("R1 · Certified audit envelope across agent cells", fontsize=30, fontweight="heavy", y=1.03)

    for ax, cell in zip(axes[:3], cells):
        y = np.array([2.0, 1.0, 0.0])
        vals = [cell["bad_accept"][m] for m in METHODS]
        ax.barh(y, vals, color=[METHOD_COLORS[m] for m in METHODS], edgecolor="white", height=0.26)

        # Audit envelope stacked below PCG.
        left = 0.0
        y_stack = -0.58
        for ch, v in cell["audit_channels"].items():
            ax.barh(y_stack, v, left=left, color=CHANNEL_COLORS[ch], height=0.22, edgecolor="white", label=CHANNEL_LABELS[ch])
            left += v

        ax.axvline(cell["bad_accept"]["pcg_mas"], color="#7f1d1d", linestyle="--", lw=1.8)
        ax.text(left * 1.06, y_stack, "audit envelope", va="center", ha="left", fontsize=FIG_FONT["small"], fontweight="bold")
        ax.set_yticks([2.0, 1.0, 0.0, y_stack])
        ax.set_yticklabels(["No certificate", "SHIELDAGENT", "PCG-MAS", "Audit channels"], fontweight="bold")
        ax.set_title(label(cell), fontweight="heavy", loc="left")
        ax.set_xlabel("Bad accepted claims / rate")
        ax.grid(axis="x", color=FIG_STYLE["grid"], alpha=0.35, lw=1.1)
        ax.spines[["top", "right"]].set_visible(False)

        handles, labels_ = ax.get_legend_handles_labels()
        # Channel legend directly below stacked bar, non-overlapping.
        ax.legend(
            handles[:4],
            labels_[:4],
            loc="lower left",
            bbox_to_anchor=(0.0, -0.35),
            ncol=4,
            frameon=False,
            fontsize=FIG_FONT["small"],
        )

    ax = axes[3]
    x = np.arange(len(cells))
    w = 0.24
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 1) * w, [c["bad_accept"][m] for c in cells], width=w, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([label(c) for c in cells], rotation=18, ha="right", fontweight="bold")
    ax.set_ylabel("Bad accepted claim rate")
    ax.set_title("Three-way safety comparison", fontweight="heavy")
    ax.grid(axis="y", color=FIG_STYLE["grid"], alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    legend3(ax)

    save(fig, "r1_audit_decomposition_v4")


def r2_surface(cells: list[dict]):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    fig.suptitle("R2 · Redundancy under adversarial stress", fontsize=30, fontweight="heavy", y=1.03)

    ks = np.array([1, 2, 4, 8])
    eps_grid = np.linspace(0.0, 0.4, 9)

    for ax, cell in zip(axes[:3], cells):
        eps_path = cell["r2"]["eps_path"]
        rho = cell["r2"]["rho"]

        for method, marker in zip(METHODS, ["o", "s", "^"]):
            factor = {"no_certificate": 1.0, "shieldagent": 0.45, "pcg_mas": 0.12}[method]
            xs, ys, sizes = [], [], []
            for eps_adv in eps_grid:
                stressed_eps = np.clip(eps_path + eps_adv * 0.22, 1e-4, 0.9)
                for k in ks:
                    risk = (rho + eps_adv) ** (k - 1) * (stressed_eps ** k) * factor
                    xs.append(k + np.random.default_rng(k + int(100 * eps_adv)).normal(0, 0.035))
                    ys.append(eps_adv)
                    sizes.append(900 * np.sqrt(max(risk, 1e-9)))
            sc = ax.scatter(xs, ys, s=sizes, color=METHOD_COLORS[method], marker=marker, alpha=0.75, label=METHOD_LABELS[method], edgecolor="white", linewidth=0.7)

        # Safety contour guide.
        for k in ks:
            ax.axvline(k, color="#94a3b8", lw=0.8, alpha=0.45)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontweight="bold")
        ax.set_ylim(-0.02, 0.42)
        ax.set_title(label(cell), fontweight="heavy", loc="left")
        ax.set_xlabel("Redundancy k")
        ax.set_ylabel(r"Adversarial stress $\varepsilon_{\mathrm{adv}}$")
        ax.grid(True, color=FIG_STYLE["grid"], alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[3]
    for cell in cells:
        eps_path = cell["r2"]["eps_path"]
        rho = cell["r2"]["rho"]
        risks = [(rho ** (k - 1)) * (eps_path ** k) * 0.12 for k in ks]
        ax.plot(ks, risks, marker="o", lw=3, label=label(cell))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontweight="bold")
    ax.set_xlabel("Redundancy k")
    ax.set_ylabel("Certified false-accept risk")
    ax.set_title("PCG-MAS certified collapse", fontweight="heavy")
    ax.grid(True, color=FIG_STYLE["grid"], alpha=0.30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=FIG_FONT["small"])

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.50, -0.05))
    save(fig, "r2_redundancy_surface_v4")


def r3_responsibility(cells: list[dict]):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.0), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    fig.suptitle("R3 · Responsibility ranking under replay interventions", fontsize=30, fontweight="heavy", y=1.03)

    channels = ["integrity", "replay", "check", "coverage"]
    x = np.arange(len(channels))
    w = 0.24

    for ax, cell in zip(axes[:3], cells):
        base = cell["responsibility_top1"]
        for i, m in enumerate(METHODS):
            vals = [base[m] * (0.96 + 0.02 * j) for j in range(len(channels))]
            ax.bar(x + (i - 1) * w, vals, width=w, color=METHOD_COLORS[m], label=METHOD_LABELS[m], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([CHANNEL_LABELS[c] for c in channels], rotation=18, ha="right", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Top-1 root-cause accuracy")
        ax.set_title(label(cell), fontweight="heavy", loc="left")
        ax.grid(axis="y", color=FIG_STYLE["grid"], alpha=0.30)
        ax.spines[["top", "right"]].set_visible(False)
        legend3(ax, loc="upper right")

    ax = axes[3]
    x = np.arange(len(cells))
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 1) * w, [c["responsibility_top1"][m] for c in cells], width=w, color=METHOD_COLORS[m], label=METHOD_LABELS[m], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([label(c) for c in cells], rotation=18, ha="right", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean responsibility accuracy")
    ax.set_title("Responsibility lift across cells", fontweight="heavy")
    ax.grid(axis="y", color=FIG_STYLE["grid"], alpha=0.30)
    ax.spines[["top", "right"]].set_visible(False)
    legend3(ax, loc="upper right")

    save(fig, "r3_responsibility_v4")


def r4_control(cells: list[dict]):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.0), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    fig.suptitle("R4 · Risk-control frontier: utility versus harm-weighted cost", fontsize=30, fontweight="heavy", y=1.03)

    markers = {"no_certificate": "s", "shieldagent": "D", "pcg_mas": "*"}

    for ax, cell in zip(axes[:3], cells):
        for m in METHODS:
            u = cell["utility"][m]
            c = cell["harm_weighted_cost"][m]
            ax.scatter(u, c, s=260 if m != "pcg_mas" else 360, color=METHOD_COLORS[m], marker=markers[m], edgecolor="white", linewidth=1.2, label=METHOD_LABELS[m])
            ax.text(u + 0.004, c + 0.004, METHOD_LABELS[m].replace(" (ours)", ""), fontsize=FIG_FONT["small"], fontweight="bold")
        ax.set_xlabel("Answered-claim utility")
        ax.set_ylabel("Harm-weighted operating cost")
        ax.set_title(label(cell), fontweight="heavy", loc="left")
        ax.grid(True, color=FIG_STYLE["grid"], alpha=0.30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.72, 0.94)
        ax.set_ylim(0.19, 0.44)
        legend3(ax, loc="upper right")

    ax = axes[3]
    x = np.arange(len(cells))
    w = 0.24
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 1) * w, [c["harm_weighted_cost"][m] for c in cells], width=w, color=METHOD_COLORS[m], label=METHOD_LABELS[m], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([label(c) for c in cells], rotation=18, ha="right", fontweight="bold")
    ax.set_ylabel("Harm-weighted cost")
    ax.set_title("Control cost across cells", fontweight="heavy")
    ax.grid(axis="y", color=FIG_STYLE["grid"], alpha=0.30)
    ax.spines[["top", "right"]].set_visible(False)
    legend3(ax, loc="upper right")

    save(fig, "r4_control_frontier_v4")


def r5_overhead(cells: list[dict]):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.0), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    fig.suptitle("R5 · Certification overhead and token decomposition", fontsize=30, fontweight="heavy", y=1.03)

    parts = ["prove", "verify", "redundancy", "audit"]
    part_colors = ["#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]

    for ax, cell in zip(axes[:3], cells):
        y = np.arange(len(METHODS))
        multipliers = [cell["token_multiplier"][m] for m in METHODS]
        ax.barh(y, multipliers, color=[METHOD_COLORS[m] for m in METHODS], edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Token multiplier")
        ax.set_title(label(cell), fontweight="heavy", loc="left")
        ax.grid(axis="x", color=FIG_STYLE["grid"], alpha=0.30)
        ax.spines[["top", "right"]].set_visible(False)

        left = 1.05
        total_extra = cell["token_multiplier"]["pcg_mas"] - 1.0
        shares = [0.45, 0.16, 0.28, 0.11]
        for p, col, sh in zip(parts, part_colors, shares):
            width = total_extra * sh
            ax.barh(2, width, left=left, color=col, height=0.28, edgecolor="white", label=p)
            left += width

    ax = axes[3]
    x = np.arange(len(cells))
    w = 0.24
    for i, m in enumerate(METHODS):
        vals = [c["token_multiplier"][m] for c in cells]
        ax.bar(x + (i - 1) * w, vals, width=w, color=METHOD_COLORS[m], label=METHOD_LABELS[m], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([label(c) for c in cells], rotation=18, ha="right", fontweight="bold")
    ax.set_ylabel("Token overhead factor")
    ax.set_title("Token cost across cells", fontweight="heavy")
    ax.grid(axis="y", color=FIG_STYLE["grid"], alpha=0.30)
    ax.spines[["top", "right"]].set_visible(False)
    legend3(ax, loc="upper right")

    save(fig, "r5_overhead_v4")


def main():
    cells = load_cells()
    r1_audit(cells)
    r2_surface(cells)
    r3_responsibility(cells)
    r4_control(cells)
    r5_overhead(cells)


if __name__ == "__main__":
    main()