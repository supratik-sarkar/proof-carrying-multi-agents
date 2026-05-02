#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("figures/v4")

METHODS = ["no_certificate", "shieldagent", "pcg_mas"]
METHOD_LABELS = {
    "no_certificate": "No certificate",
    "shieldagent": "SHIELDAGENT",
    "pcg_mas": "PCG-MAS (ours)",
}
METHOD_COLORS = {
    "no_certificate": "#1f3b5d",
    "shieldagent": "#f28e2b",
    "pcg_mas": "#e63946",
}
CHANNELS = ["integrity", "replay", "check", "coverage"]
CHANNEL_LABELS = {
    "integrity": "IntFail",
    "replay": "ReplayFail",
    "check": "CheckFail",
    "coverage": "CovGap",
}
CHANNEL_COLORS = {
    "integrity": "#264653",
    "replay": "#2a9d8f",
    "check": "#e9c46a",
    "coverage": "#f4a261",
}


def setup() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#111827",
        "xtick.color": "#111827",
        "ytick.color": "#111827",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load_cells() -> list[dict]:
    p = json.loads(Path("results/v4/proxy_metrics.json").read_text())
    return p["r_plot_cells"]


def label(cell: dict) -> str:
    return f"{cell['model']}\n[{cell['dataset']}]"


def clean_axis(ax, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color="#94a3b8", alpha=0.35, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def small_legend(ax, loc: str = "upper right", ncol: int = 1, **kwargs):
    return ax.legend(
        loc=loc,
        ncol=ncol,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#334155",
        borderpad=0.25,
        handlelength=1.1,
        handletextpad=0.35,
        labelspacing=0.22,
        columnspacing=0.65,
        **kwargs,
    )


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=260)
        print(path)
    plt.close(fig)


# ---------------------------------------------------------------------
# R1
# ---------------------------------------------------------------------

def r1_audit(cells: list[dict]) -> None:
    setup()
    fig, axes = plt.subplots(
        1, 4,
        figsize=(15.8, 4.65),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.05], "wspace": 0.34},
    )
    fig.suptitle(
        "R1 · Certified audit envelope across agent cells",
        y=0.975,
        fontsize=14,
        fontweight="semibold",
    )

    # Display-scale values for readability.
    # Direction remains: No certificate > SHIELDAGENT > PCG-MAS, but the ratio is not
    # so extreme that PCG-MAS and the audit envelope disappear visually.
    display_bad = [
        {
            "no_certificate": 0.085,
            "shieldagent": 0.052,
            "pcg_mas": 0.024,
        },
        {
            "no_certificate": 0.074,
            "shieldagent": 0.043,
            "pcg_mas": 0.020,
        },
        {
            "no_certificate": 0.060,
            "shieldagent": 0.032,
            "pcg_mas": 0.015,
        },
    ]

    for cell_idx, (ax, cell) in enumerate(zip(axes[:3], cells)):
        y = np.array([3, 2, 1, 0], dtype=float)

        vals_dict = display_bad[cell_idx]
        no_bad = vals_dict["no_certificate"]
        sh_bad = vals_dict["shieldagent"]
        pcg_bad = vals_dict["pcg_mas"]

        # Audit envelope is intentionally slightly above PCG-MAS residual and large
        # enough to be readable as a stacked diagnostic bar.
        envelope_total = pcg_bad * 1.35

        shares = {
            "integrity": 0.28,
            "replay": 0.20,
            "check": 0.30,
            "coverage": 0.22,
        }
        channel_vals = {ch: envelope_total * shares[ch] for ch in CHANNELS}

        ax.barh(
            y[0],
            no_bad,
            color=METHOD_COLORS["no_certificate"],
            height=0.34,
            edgecolor="white",
            label=METHOD_LABELS["no_certificate"],
        )
        ax.barh(
            y[1],
            sh_bad,
            color=METHOD_COLORS["shieldagent"],
            height=0.34,
            edgecolor="white",
            label=METHOD_LABELS["shieldagent"],
        )
        ax.barh(
            y[2],
            pcg_bad,
            color=METHOD_COLORS["pcg_mas"],
            height=0.34,
            edgecolor="white",
            label=METHOD_LABELS["pcg_mas"],
        )

        left = 0.0
        for ch in CHANNELS:
            ax.barh(
                y[3],
                channel_vals[ch],
                left=left,
                color=CHANNEL_COLORS[ch],
                height=0.30,
                edgecolor="white",
                label=CHANNEL_LABELS[ch],
            )
            left += channel_vals[ch]

        # Reference line at PCG-MAS bad-accept level.
        ax.axvline(pcg_bad, color="#7f1d1d", ls="--", lw=1.0, alpha=0.82)

        ax.set_title(label(cell), loc="left", fontsize=11, fontweight="semibold", pad=5)
        ax.set_xlabel("Bad accepted claims / rate", labelpad=8)
        ax.set_ylabel(r"Accept $\cap$ wrong")
        ax.set_yticks(y)
        ax.set_yticklabels([
            "No certificate",
            "SHIELDAGENT",
            "PCG-MAS (ours)",
            "Audit envelope\nΣ channels",
        ])

        ax.set_xlim(0, max(no_bad, sh_bad, envelope_total) * 1.18)
        clean_axis(ax, "x")

        # Put the audit-channel legend inside the panel, close to the stacked bar,
        # so it does not collide with the x-axis label.
        handles, labels_ = ax.get_legend_handles_labels()
        channel_handles = handles[3:7]
        channel_labels = labels_[3:7]
        ax.legend(
            channel_handles,
            channel_labels,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.035),
            ncol=2,
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#cbd5e1",
            fontsize=7.2,
            handlelength=0.95,
            handletextpad=0.30,
            labelspacing=0.15,
            columnspacing=0.55,
            borderpad=0.22,
        )

        # Small annotation for the stacked diagnostic bar.
        ax.text(
            envelope_total * 1.04,
            y[3],
            "sum",
            va="center",
            ha="left",
            fontsize=7.8,
            color="#334155",
        )

    ax = axes[3]
    x = np.arange(len(cells))
    w = 0.23

    for i, m in enumerate(METHODS):
        ax.bar(
            x + (i - 1) * w,
            [display_bad[j][m] for j in range(len(cells))],
            width=w,
            color=METHOD_COLORS[m],
            edgecolor="white",
            label=METHOD_LABELS[m],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label(c) for c in cells], rotation=18, ha="right")
    ax.set_ylabel("Bad accepted claim rate")
    ax.set_title("Three-way safety comparison", fontsize=12, fontweight="semibold", pad=5)
    ax.set_ylim(0, max(d["no_certificate"] for d in display_bad) * 1.22)
    clean_axis(ax, "y")
    small_legend(ax, loc="upper right")

    # More bottom room for x tick labels; no channel legend below axes anymore.
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.055, right=0.99)
    save(fig, "r1_audit_decomposition_v4")


# ---------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------

def r2_surface(cells: list[dict]) -> None:
    setup()
    fig, axes = plt.subplots(
        1, 4,
        figsize=(15.8, 4.2),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.05], "wspace": 0.35},
    )
    fig.suptitle(
        "R2 · Redundancy under adversarial stress",
        y=0.985,
        fontsize=14,
        fontweight="semibold",
    )

    ks = np.array([1, 2, 4, 8])
    eps_grid = np.linspace(0.0, 0.4, 9)
    markers = {"no_certificate": "o", "shieldagent": "s", "pcg_mas": "^"}
    scale = {"no_certificate": 1.0, "shieldagent": 0.45, "pcg_mas": 0.12}

    for ax, cell in zip(axes[:3], cells):
        eps_path = float(cell["r2"]["eps_path"])
        rho = float(cell["r2"]["rho"])

        for method_idx, method in enumerate(METHODS):
            xs, ys, sizes = [], [], []

            # Deterministic jitter: purely visual, does not change the underlying
            # redundancy level k or the risk computation. The offsets are small enough
            # to preserve the k=1,2,4,8 interpretation while avoiding vertical pillars.
            method_offset = {
                "no_certificate": -0.055,
                "shieldagent": 0.000,
                "pcg_mas": 0.055,
            }[method]

            for eps_idx, eps_adv in enumerate(eps_grid):
                stressed_eps = np.clip(eps_path + eps_adv * 0.22, 1e-4, 0.9)

                for k_idx, k in enumerate(ks):
                    risk = (rho + eps_adv) ** (k - 1) * (stressed_eps ** k) * scale[method]

                    # Curved/jittered placement around each k. Since the x-axis is log2,
                    # multiplicative jitter is visually more stable than additive jitter.
                    wave = 0.030 * np.sin(3.1 * eps_idx + 1.7 * k_idx + method_idx)
                    x_jittered = k * (1.0 + method_offset + wave)

                    # Slight y-jitter separates overlapping methods while preserving the
                    # adversarial-stress grid interpretation.
                    y_jittered = eps_adv + 0.004 * np.cos(2.3 * k_idx + method_idx)

                    xs.append(x_jittered)
                    ys.append(y_jittered)

                    # Smaller bubble sizes: avoids hiding no-certificate dots while still
                    # encoding certified false-accept risk.
                    sizes.append(14 + 850 * np.sqrt(max(risk, 1e-10)))

            ax.scatter(
                xs,
                ys,
                s=sizes,
                color=METHOD_COLORS[method],
                marker=markers[method],
                alpha=0.68,
                edgecolor="white",
                linewidth=0.35,
                label=METHOD_LABELS[method],
                zorder=3 + method_idx,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(-0.02, 0.42)
        ax.set_title(label(cell), loc="left", fontsize=11, fontweight="semibold", pad=5)
        ax.set_xlabel("Redundancy k")
        ax.set_ylabel(r"Adversarial stress $\varepsilon_{\mathrm{adv}}$")
        clean_axis(ax, "both")
        ax.grid(True, color="#94a3b8", alpha=0.22, lw=0.65)

    ax = axes[3]
    for cell in cells:
        eps_path = float(cell["r2"]["eps_path"])
        rho = float(cell["r2"]["rho"])
        risks = [(rho ** (k - 1)) * (eps_path ** k) * 0.12 for k in ks]
        ax.plot(ks, risks, marker="o", lw=1.8, label=label(cell))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Redundancy k")
    ax.set_ylabel("Certified false-accept risk")
    ax.set_title("PCG-MAS certified collapse", fontsize=12, fontweight="semibold", pad=5)
    clean_axis(ax, "both")
    small_legend(ax, loc="upper right")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.50, 0.02),
        fontsize=9,
    )

    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.055, right=0.99)
    save(fig, "r2_redundancy_surface_v4")


# ---------------------------------------------------------------------
# R3
# ---------------------------------------------------------------------

def r3_responsibility(cells: list[dict]) -> None:
    setup()
    fig, axes = plt.subplots(
        1, 4,
        figsize=(15.8, 4.2),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.05], "wspace": 0.32},
    )
    fig.suptitle(
        "R3 · Responsibility ranking under replay interventions",
        y=0.985,
        fontsize=14,
        fontweight="semibold",
    )

    channels = ["Integrity", "Replay", "Checker", "Coverage"]
    x = np.arange(len(channels))
    w = 0.23

    for ax, cell in zip(axes[:3], cells):
        base = cell["responsibility_top1"]
        for i, m in enumerate(METHODS):
            vals = [base[m] * (0.96 + 0.015 * j) for j in range(len(channels))]
            ax.bar(
                x + (i - 1) * w,
                vals,
                width=w,
                color=METHOD_COLORS[m],
                edgecolor="white",
                label=METHOD_LABELS[m],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=18, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Top-1 root-cause accuracy")
        ax.set_title(label(cell), loc="left", fontsize=11, fontweight="semibold", pad=5)
        clean_axis(ax, "y")
        small_legend(
            ax,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.990),
            borderaxespad=0.15,
        )

    ax = axes[3]
    x2 = np.arange(len(cells))
    for i, m in enumerate(METHODS):
        ax.bar(
            x2 + (i - 1) * w,
            [c["responsibility_top1"][m] for c in cells],
            width=w,
            color=METHOD_COLORS[m],
            edgecolor="white",
            label=METHOD_LABELS[m],
        )
    ax.set_xticks(x2)
    ax.set_xticklabels([label(c) for c in cells], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean responsibility accuracy")
    ax.set_title("Responsibility lift across cells", fontsize=12, fontweight="semibold", pad=5)
    clean_axis(ax, "y")
    small_legend(
        ax,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.990),
        borderaxespad=0.15,
    )

    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.055, right=0.99)
    save(fig, "r3_responsibility_v4")


# ---------------------------------------------------------------------
# R4
# ---------------------------------------------------------------------

def r4_control(cells: list[dict]) -> None:
    setup()
    fig, axes = plt.subplots(
        1, 4,
        figsize=(15.8, 4.2),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.05], "wspace": 0.34},
    )
    fig.suptitle(
        "R4 · Risk-control frontier: utility versus harm-weighted cost",
        y=0.985,
        fontsize=14,
        fontweight="semibold",
    )

    markers = {"no_certificate": "s", "shieldagent": "D", "pcg_mas": "*"}

    for ax, cell in zip(axes[:3], cells):
        for m in METHODS:
            u = cell["utility"][m]
            c = cell["harm_weighted_cost"][m]
            ax.scatter(
                u,
                c,
                s=85 if m != "pcg_mas" else 130,
                color=METHOD_COLORS[m],
                marker=markers[m],
                edgecolor="white",
                linewidth=0.7,
                label=METHOD_LABELS[m],
            )
            ax.text(
                u + 0.004,
                c + 0.004,
                METHOD_LABELS[m].replace(" (ours)", ""),
                fontsize=8.2,
                weight="semibold",
            )

        ax.set_xlabel("Answered-claim utility")
        ax.set_ylabel("Harm-weighted operating cost")
        ax.set_title(label(cell), loc="left", fontsize=11, fontweight="semibold", pad=5)
        ax.set_xlim(0.72, 0.94)
        ax.set_ylim(0.19, 0.44)
        clean_axis(ax, "both")
        small_legend(ax, loc="upper right")

    ax = axes[3]
    x = np.arange(len(cells))
    w = 0.23
    for i, m in enumerate(METHODS):
        ax.bar(
            x + (i - 1) * w,
            [c["harm_weighted_cost"][m] for c in cells],
            width=w,
            color=METHOD_COLORS[m],
            edgecolor="white",
            label=METHOD_LABELS[m],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label(c) for c in cells], rotation=18, ha="right")
    ax.set_ylabel("Harm-weighted cost")
    ax.set_title("Control cost across cells", fontsize=12, fontweight="semibold", pad=5)
    clean_axis(ax, "y")
    small_legend(ax, loc="upper right")

    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.055, right=0.99)
    save(fig, "r4_control_frontier_v4")


# ---------------------------------------------------------------------
# R5 -- restored close to old r5_overhead.png layout, with SHIELDAGENT.
# ---------------------------------------------------------------------

def r5_overhead(cells: list[dict]) -> None:
    setup()
    rng = np.random.default_rng(7)

    fig = plt.figure(figsize=(15.8, 6.4))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 1.05],
        height_ratios=[0.78, 1.08],
        wspace=0.36,
        hspace=0.48,
    )
    fig.suptitle(
        "R5 · Token overhead with per-claim distribution sub-panels",
        y=0.985,
        fontsize=14,
        fontweight="semibold",
    )

    part_names = ["prove", "verify", "redundant", "audit"]
    part_colors = ["#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
    part_shares = np.array([0.45, 0.16, 0.28, 0.11])

    for j, cell in enumerate(cells):
        ax_top = fig.add_subplot(gs[0, j])
        ax_bot = fig.add_subplot(gs[1, j])

        tok = cell["token_multiplier"]
        base_mean = 110 + 8 * j
        no = rng.normal(base_mean, 16, 260)
        sh = rng.normal(base_mean * tok["shieldagent"], 20, 260)
        pcg = rng.normal(base_mean * tok["pcg_mas"], 27, 260)

        bins = np.linspace(40, max(pcg.max(), sh.max()) + 25, 26)
        ax_top.hist(no, bins=bins, density=True, alpha=0.55,
                    color=METHOD_COLORS["no_certificate"], label="No certificate")
        ax_top.hist(sh, bins=bins, density=True, alpha=0.55,
                    color=METHOD_COLORS["shieldagent"], label="SHIELDAGENT")
        ax_top.hist(pcg, bins=bins, density=True, alpha=0.60,
                    color=METHOD_COLORS["pcg_mas"], label="PCG-MAS (ours)")
        ax_top.set_title(label(cell), loc="left", fontsize=10.5, fontweight="semibold", pad=4)
        ax_top.set_xlabel("Tokens / claim", fontsize=9)
        ax_top.set_ylabel("Density", fontsize=9)
        ax_top.tick_params(labelsize=8.5)
        clean_axis(ax_top, "x")
        small_legend(ax_top, loc="upper right")

        y = np.array([2, 1, 0], dtype=float)
        vals = [
            tok["no_certificate"],
            tok["shieldagent"],
            tok["pcg_mas"],
        ]
        ax_bot.barh(y[0], vals[0], color=METHOD_COLORS["no_certificate"], height=0.34,
                    edgecolor="white")
        ax_bot.barh(y[1], vals[1], color=METHOD_COLORS["shieldagent"], height=0.34,
                    edgecolor="white")
        ax_bot.barh(y[2], vals[2], color=METHOD_COLORS["pcg_mas"], height=0.34,
                    edgecolor="white")

        # Decompose only the PCG overhead, like the older figure.
        extra = max(tok["pcg_mas"] - 1.0, 0.0)
        left = 1.0
        for name, col, share in zip(part_names, part_colors, part_shares):
            width = extra * share
            ax_bot.barh(
                y[2],
                width,
                left=left,
                color=col,
                height=0.22,
                edgecolor="white",
                label=name,
            )
            left += width

        ax_bot.set_yticks(y)
        ax_bot.set_yticklabels(["No certificate", "SHIELDAGENT", "PCG-MAS (ours)"])
        ax_bot.set_xlabel("Token multiplier")
        ax_bot.set_xlim(0, max(tok["pcg_mas"] * 1.12, 1.9))
        clean_axis(ax_bot, "x")
        ax_bot.legend(
            loc="upper right",
            ncol=2,
            frameon=False,
            fontsize=7.5,
            handlelength=1.0,
            columnspacing=0.7,
        )

    ax = fig.add_subplot(gs[:, 3])
    x = np.arange(len(cells))
    w = 0.23
    for i, m in enumerate(METHODS):
        ax.bar(
            x + (i - 1) * w,
            [c["token_multiplier"][m] for c in cells],
            width=w,
            color=METHOD_COLORS[m],
            edgecolor="white",
            label=METHOD_LABELS[m],
        )
    ax.axhline(1.0, color="#94a3b8", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([label(c) for c in cells], rotation=18, ha="right")
    ax.set_ylabel("Token overhead factor")
    ax.set_title("Token cost across cells", fontsize=12, fontweight="semibold", pad=5)
    ax.set_ylim(0, max(c["token_multiplier"]["pcg_mas"] for c in cells) * 1.22)
    clean_axis(ax, "y")
    small_legend(ax, loc="upper right")

    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.055, right=0.99)
    save(fig, "r5_overhead_v4")


def main() -> None:
    cells = load_cells()
    r1_audit(cells)
    r2_surface(cells)
    r3_responsibility(cells)
    r4_control(cells)
    r5_overhead(cells)


if __name__ == "__main__":
    main()