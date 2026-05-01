#!/usr/bin/env python3
"""Build a combined R1-R5 appendix summary figure.

The script creates a contact-sheet style figure from existing PNG/PDF/SVG files.
If a figure is missing, it leaves a labeled placeholder. This lets the pipeline run
end-to-end while individual experiments are still being refined.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


R_KEYS = [
    ("R1", "audit"),
    ("R2", "redundancy"),
    ("R3", "responsibility"),
    ("R4", "pareto"),
    ("R5", "overhead"),
]


def find_candidate(figures_dir: Path, token: str) -> Path | None:
    if not figures_dir.exists():
        return None

    candidates: list[Path] = []
    for suffix in (".png", ".jpg", ".jpeg"):
        candidates.extend(figures_dir.rglob(f"*{token}*{suffix}"))
        candidates.extend(figures_dir.rglob(f"*{token.upper()}*{suffix}"))

    return sorted(candidates)[0] if candidates else None


def draw_placeholder(ax, title: str, token: str) -> None:
    ax.add_patch(Rectangle((0.05, 0.08), 0.9, 0.84, fill=False, linewidth=1.0))
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.42, f"missing figure token: {token}", ha="center", va="center", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--output", type=Path, default=Path("figures/v4/r1_r5_summary.png"))
    parser.add_argument("--pdf-output", type=Path, default=Path("figures/v4/r1_r5_summary.pdf"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pdf_output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.flatten()

    for ax, (label, token) in zip(axes_flat, R_KEYS):
        path = find_candidate(args.figures_dir, token)
        ax.set_title(label, fontsize=12, fontweight="bold")
        if path is None:
            draw_placeholder(ax, label, token)
        else:
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(path.name, fontsize=7)

    # Sixth panel: compact interpretation.
    ax = axes_flat[-1]
    ax.axis("off")
    ax.text(
        0.0,
        0.95,
        "PCG-MAS v4 summary",
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        0.78,
        "R1: finite-sample audit envelope\n"
        "R2: redundancy-driven false-accept decay\n"
        "R3: replay-based responsibility recovery\n"
        "R4: cost--harm/privacy frontier\n"
        "R5: token and latency overhead",
        fontsize=10,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.pdf_output, bbox_inches="tight")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.pdf_output}")


if __name__ == "__main__":
    main()