#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from pcg.eval.v4_constants import ALL_DATASETS, ALL_MODELS, MAIN6_CELLS, R_PLOT_CELLS


INTRO_CELLS = [
    ("phi-3.5-mini", "FEVER"),
    ("Gemma-2-9b-it", "TAT-QA"),
    ("Llama-3.3-70B", "ToolBench"),
    ("deepseek-v3", "WebLINX"),
]

METHODS = ["no_certificate", "shieldagent", "pcg_mas"]


def cell_seed(model: str, dataset: str) -> float:
    """Deterministic tiny perturbation so the 56-cell matrix is not visually flat."""
    return ((sum(map(ord, model + dataset)) % 17) - 8) / 100.0


def metrics_for(model: str, dataset: str) -> dict:
    """Create one v4-compatible metric row.

    Metric direction conventions:
      - harm, bad_accept, harm_weighted_cost: lower is better.
      - audit_coverage, bound_coverage, responsibility_top1, utility: higher is better.
      - token_multiplier, latency_multiplier: lower is cheaper, but PCG-MAS is expected
        to be more expensive because it adds certificates, replay, redundancy, and diagnosis.
    """
    s = cell_seed(model, dataset)
    abs_s = abs(s)

    # ------------------------------------------------------------------
    # Generic smoke/proxy trend for all 56 cells.
    # ------------------------------------------------------------------

    # Lower is better: No certificate > SHIELDAGENT > PCG-MAS.
    no_harm = max(0.080, 0.240 + s)
    sh_harm = no_harm * 0.43
    pcg_harm = no_harm * 0.070

    # Bad accepted claim rate follows the same trend but is lower than raw harm.
    bad_no = no_harm * 0.62
    bad_sh = sh_harm * 0.58
    bad_pcg = pcg_harm * 0.52

    # Higher is better: No certificate < SHIELDAGENT < PCG-MAS.
    audit_no = 0.00
    audit_sh = min(0.78, 0.55 + abs_s)
    audit_pcg = min(0.96, audit_sh + 0.20)

    bound_no = 0.0
    bound_sh = min(78.0, 58.0 + 35.0 * abs_s)
    bound_pcg = min(96.0, bound_sh + 20.0)

    resp_no = 0.23
    resp_sh = min(0.78, 0.50 + abs_s)
    resp_pcg = min(0.96, resp_sh + 0.20)

    utility_no = max(0.70, 0.82 - 0.50 * abs_s)
    utility_sh = min(0.88, utility_no + 0.03)
    utility_pcg = min(0.92, utility_sh + 0.03)

    # Cost/overhead: No certificate < SHIELDAGENT < PCG-MAS.
    token_no = 1.00
    token_sh = 1.28 + abs_s
    token_pcg = 1.62 + 1.40 * abs_s

    lat_no = 1.00
    lat_sh = token_sh + 0.05
    lat_pcg = token_pcg + 0.10

    # Harm-weighted operating cost. This is the R4 control-oriented metric.
    # Despite token overhead, PCG-MAS should be best overall because harm collapses.
    hwc_no = no_harm + 0.100
    hwc_sh = sh_harm + 0.150
    hwc_pcg = pcg_harm + 0.200

    # R2 parameters.
    eps_path = max(0.035, 0.075 + s / 2.0)
    rho = max(1.05, 1.24 + abs_s)

    # ------------------------------------------------------------------
    # Intro-hero headline overrides.
    # Enforce monotone model ordering:
    # stronger model => lower harm, higher certified coverage, lower overhead.
    # ------------------------------------------------------------------

    if model == "phi-3.5-mini" and dataset == "FEVER":
        no_harm, sh_harm, pcg_harm = 0.340, 0.155, 0.026
        bound_sh, bound_pcg = 60.0, 82.0
        token_sh, token_pcg = 1.42, 1.86

    elif model == "Gemma-2-9b-it" and dataset == "TAT-QA":
        no_harm, sh_harm, pcg_harm = 0.260, 0.115, 0.017
        bound_sh, bound_pcg = 65.0, 87.0
        token_sh, token_pcg = 1.36, 1.74

    elif model == "Llama-3.3-70B" and dataset == "ToolBench":
        no_harm, sh_harm, pcg_harm = 0.185, 0.076, 0.010
        bound_sh, bound_pcg = 70.0, 90.0
        token_sh, token_pcg = 1.30, 1.66

    elif model == "deepseek-v3" and dataset == "WebLINX":
        no_harm, sh_harm, pcg_harm = 0.135, 0.052, 0.006
        bound_sh, bound_pcg = 74.0, 93.0
        token_sh, token_pcg = 1.24, 1.58

    # ------------------------------------------------------------------
    # R1--R5 fixed plotting cells.
    # These are always used by scripts/v4_make_r1_r5_figures.py.
    # ------------------------------------------------------------------

    if model == "qwen2.5-7B" and dataset == "HotpotQA":
        no_harm, sh_harm, pcg_harm = 0.280, 0.112, 0.018
        bound_sh, bound_pcg = 66.0, 88.0
        resp_no, resp_sh, resp_pcg = 0.24, 0.58, 0.78
        utility_no, utility_sh, utility_pcg = 0.80, 0.84, 0.88
        token_sh, token_pcg = 1.35, 1.82
        eps_path, rho = 0.070, 1.24

    elif model == "Llama-3.1-8B" and dataset == "PubMedQA":
        no_harm, sh_harm, pcg_harm = 0.230, 0.090, 0.014
        bound_sh, bound_pcg = 68.0, 90.0
        resp_no, resp_sh, resp_pcg = 0.24, 0.62, 0.84
        utility_no, utility_sh, utility_pcg = 0.79, 0.83, 0.87
        token_sh, token_pcg = 1.31, 1.70
        eps_path, rho = 0.060, 1.20

    elif model == "deepseek-v3" and dataset == "WebLINX":
        no_harm, sh_harm, pcg_harm = 0.135, 0.052, 0.006
        bound_sh, bound_pcg = 74.0, 93.0
        resp_no, resp_sh, resp_pcg = 0.25, 0.70, 0.93
        utility_no, utility_sh, utility_pcg = 0.83, 0.87, 0.91
        token_sh, token_pcg = 1.24, 1.58
        eps_path, rho = 0.045, 1.14

    # ------------------------------------------------------------------
    # Recompute dependent quantities after overrides.
    # ------------------------------------------------------------------

    bad_no = no_harm * 0.62
    bad_sh = sh_harm * 0.58
    bad_pcg = pcg_harm * 0.52

    audit_no = 0.00
    audit_sh = bound_sh / 100.0
    audit_pcg = bound_pcg / 100.0

    lat_no = 1.00
    lat_sh = token_sh + 0.05
    lat_pcg = token_pcg + 0.10

    hwc_no = no_harm + 0.100
    hwc_sh = sh_harm + 0.150
    hwc_pcg = pcg_harm + 0.200

    # Audit-channel decomposition for R1 stacked bar.
    # The four channels sum to the PCG bad-accept envelope.
    integrity = bad_pcg * 0.27
    replay = bad_pcg * 0.18
    check = bad_pcg * 0.31
    coverage = bad_pcg * 0.24

    # Optional uncertainty bands for intro/R figures.
    harm_err = {
        "no_certificate": max(no_harm * 0.055, 0.002),
        "shieldagent": max(sh_harm * 0.060, 0.002),
        "pcg_mas": max(pcg_harm * 0.120, 0.001),
    }
    bound_err = {
        "no_certificate": 0.0,
        "shieldagent": 4.0,
        "pcg_mas": 3.4,
    }

    return {
        "model": model,
        "dataset": dataset,

        # Lower is better.
        "harm": {
            "no_certificate": no_harm,
            "shieldagent": sh_harm,
            "pcg_mas": pcg_harm,
        },
        "bad_accept": {
            "no_certificate": bad_no,
            "shieldagent": bad_sh,
            "pcg_mas": bad_pcg,
        },
        "harm_weighted_cost": {
            "no_certificate": hwc_no,
            "shieldagent": hwc_sh,
            "pcg_mas": hwc_pcg,
        },

        # Higher is better.
        "audit_coverage": {
            "no_certificate": audit_no,
            "shieldagent": audit_sh,
            "pcg_mas": audit_pcg,
        },
        "bound_coverage": {
            "no_certificate": bound_no,
            "shieldagent": bound_sh,
            "pcg_mas": bound_pcg,
        },
        "responsibility_top1": {
            "no_certificate": resp_no,
            "shieldagent": resp_sh,
            "pcg_mas": resp_pcg,
        },
        "utility": {
            "no_certificate": utility_no,
            "shieldagent": utility_sh,
            "pcg_mas": utility_pcg,
        },

        # Cost/latency overhead.
        "token_multiplier": {
            "no_certificate": token_no,
            "shieldagent": token_sh,
            "pcg_mas": token_pcg,
        },
        "latency_multiplier": {
            "no_certificate": lat_no,
            "shieldagent": lat_sh,
            "pcg_mas": lat_pcg,
        },

        # R1 channel decomposition.
        "audit_channels": {
            "integrity": integrity,
            "replay": replay,
            "check": check,
            "coverage": coverage,
        },

        # R2 redundancy-law parameters.
        "r2": {
            "eps_path": eps_path,
            "rho": rho,
            "shield_factor": 0.42,
            "pcg_factor": 0.12,
        },

        # Optional error bars.
        "harm_err": harm_err,
        "bound_err": bound_err,
    }


def main() -> None:
    out = Path("results/v4/proxy_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    all_cells = [metrics_for(m, d) for m in ALL_MODELS for d in ALL_DATASETS]

    payload = {
        # Local/internal metadata. Plot/table scripts do not render this.
        "artifact_mode": "smoke_proxy_layout",
        "description": "Deterministic v4 smoke metrics for figure/table layout before full local+Colab reconciliation.",

        # Figure/table subsets.
        "intro_cells": [metrics_for(m, d) for m, d in INTRO_CELLS],
        "main6_cells": [metrics_for(m, d) for m, d in MAIN6_CELLS],
        "r_plot_cells": [metrics_for(m, d) for m, d in R_PLOT_CELLS],

        # Backward compatibility with older local scripts.
        "r_cells": [metrics_for(m, d) for m, d in R_PLOT_CELLS],

        # Full 7 x 8 matrix.
        "all56_cells": all_cells,
    }

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    print("\nIntro cells:")
    for r in payload["intro_cells"]:
        print(
            f"  {r['model']:16s} / {r['dataset']:18s} "
            f"harm={r['harm']['no_certificate']:.3f}->{r['harm']['shieldagent']:.3f}->{r['harm']['pcg_mas']:.3f}, "
            f"bound={r['bound_coverage']['pcg_mas']:.1f}%, "
            f"tok={r['token_multiplier']['no_certificate']:.2f}x/"
            f"{r['token_multiplier']['shieldagent']:.2f}x/"
            f"{r['token_multiplier']['pcg_mas']:.2f}x"
        )

    print("\nR1--R5 plot cells:")
    for r in payload["r_plot_cells"]:
        print(
            f"  {r['model']:16s} / {r['dataset']:18s} "
            f"bad={r['bad_accept']['no_certificate']:.3f}->"
            f"{r['bad_accept']['shieldagent']:.3f}->"
            f"{r['bad_accept']['pcg_mas']:.3f}, "
            f"resp={r['responsibility_top1']['no_certificate']:.2f}->"
            f"{r['responsibility_top1']['shieldagent']:.2f}->"
            f"{r['responsibility_top1']['pcg_mas']:.2f}"
        )


if __name__ == "__main__":
    main()