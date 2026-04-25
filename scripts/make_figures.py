"""
scripts/make_figures.py

Regenerates ALL paper figures from saved experiment JSON files. This script
NEVER reruns experiments — it only reads results/.

Each plot function in pcg.eval.plots has a specific signature; this script
is responsible for translating the experiment-result JSON shape into the
exact arguments those functions expect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from _common import log_info, log_section, project_root


def _latest_run(results_dir: Path, rid: str) -> Path | None:
    base = results_dir / rid
    if not base.exists():
        return None
    candidates = sorted([p for p in base.iterdir() if p.is_dir()],
                         key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if (c / f"{rid}.json").exists():
            return c
    return None


def _load(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return json.load(fh)


def _safe_ci(d: dict, key: str = "ci", fallback: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Pull a CI tuple out of a dict, handling None/missing safely."""
    v = d.get(key)
    if v is None or len(v) != 2:
        return fallback
    return (float(v[0]), float(v[1]))


# ---------------------------------------------------------------------------
# Per-figure handlers — each translates JSON to the plot fn's exact signature
# ---------------------------------------------------------------------------


def _make_r1_figure(r1_run: Path, out_dir: Path, *, plot_fn, save_fn) -> str | None:
    data = _load(r1_run / "r1.json")
    agg = data["aggregated"]
    decomp = {
        "lhs_accept_and_wrong": agg["lhs_accept_and_wrong"]["mean"],
        "ci_lhs": _safe_ci(agg["lhs_accept_and_wrong"]),
        "p_int_fail": agg["p_int_fail"]["mean"],
        "ci_int_fail": _safe_ci(agg["p_int_fail"]),
        "p_replay_fail": agg["p_replay_fail"]["mean"],
        "ci_replay_fail": _safe_ci(agg["p_replay_fail"]),
        "p_check_fail": agg["p_check_fail"]["mean"],
        "ci_check_fail": _safe_ci(agg["p_check_fail"]),
        "p_cov_gap": agg["p_cov_gap"]["mean"],
        "ci_cov_gap": _safe_ci(agg["p_cov_gap"]),
    }
    fig = plot_fn(decomp=decomp)
    target = out_dir / "r1_audit_decomposition"
    save_fn(fig, str(target))
    return str(target)


def _make_r2_figure(r2_run: Path, out_dir: Path, *, plot_fn, save_fn) -> str | None:
    data = _load(r2_run / "r2.json")
    per_k = data["aggregated_per_k"]
    if not per_k:
        return None
    ks = [row["k"] for row in per_k]
    empirical = [row["empirical_mean"] for row in per_k]
    empirical_ci = [tuple(row["empirical_ci"]) for row in per_k]
    theory_curve = [row["theory_plug_in_mean"] for row in per_k]
    rho_ucb_curve = [row["theory_ucb_max"] for row in per_k]
    fig = plot_fn(
        ks=ks,
        empirical=empirical,
        empirical_ci=empirical_ci,
        theory_curve=theory_curve,
        rho_ucb_curve=rho_ucb_curve,
    )
    target = out_dir / "r2_redundancy_law"
    save_fn(fig, str(target))
    return str(target)


def _make_r3_figure(r3_run: Path, out_dir: Path, *, plot_fn, save_fn) -> str | None:
    data = _load(r3_run / "r3.json")
    agg = data["aggregated"]
    if not agg:
        return None
    regime_names = list(agg.keys())
    # Build a 1-row "matrix" since we report top-1 accuracy per regime.
    component_names = ["Top-1 root-cause accuracy"]
    row_vals = []
    halfwidths = []
    for r in regime_names:
        m = agg[r].get("top1_accuracy_mean")
        m = float(m) if m is not None else 0.0
        row_vals.append(m)
        ci_lo, ci_hi = _safe_ci(agg[r], fallback=(m, m))
        halfwidths.append((ci_hi - ci_lo) / 2.0)
    resp_matrix = np.array([row_vals], dtype=float)
    ci_halfwidth = np.array([halfwidths], dtype=float)
    fig = plot_fn(
        component_names=component_names,
        regime_names=regime_names,
        resp_matrix=resp_matrix,
        ci_halfwidth=ci_halfwidth,
    )
    target = out_dir / "r3_responsibility"
    save_fn(fig, str(target))
    return str(target)


def _make_r4_pareto_figure(r4_run: Path, out_dir: Path, *, plot_fn, save_fn) -> str | None:
    data = _load(r4_run / "r4.json")
    agg = data["aggregated"]
    if not agg:
        return None
    eps_keys = list(agg.keys())
    # Each policy gets cost/harm arrays swept over eps levels.
    policies_dict: dict[str, dict] = {}
    for pname in ["always_answer", "threshold_pcg", "learned"]:
        costs = []
        harms = []
        for ek in eps_keys:
            if pname in agg[ek]:
                costs.append(agg[ek][pname]["cost_mean"])
                harms.append(agg[ek][pname]["harm_mean"])
        if costs:
            policies_dict[pname] = {"cost": costs, "harm": harms, "label": pname}
    if not policies_dict:
        return None
    fig = plot_fn(policies=policies_dict)
    target = out_dir / "r4_risk_pareto"
    save_fn(fig, str(target))
    return str(target)


def _make_r4_privacy_figure(r4_run: Path, out_dir: Path, *, plot_fn, save_fn) -> str | None:
    data = _load(r4_run / "r4.json")
    agg = data["aggregated"]
    if not agg:
        return None
    eps_keys = list(agg.keys())
    eps_values: list[float] = []
    utility: list[float] = []
    leakage: list[float] = []
    for ek in eps_keys:
        try:
            eps_val = float("inf") if ek == "inf" else float(ek)
        except ValueError:
            continue
        if "threshold_pcg" not in agg[ek]:
            continue
        eps_values.append(eps_val)
        # utility := 1 - harm of threshold_pcg policy
        h_thresh = agg[ek]["threshold_pcg"]["harm_mean"]
        utility.append(max(0.0, 1.0 - h_thresh))
        # leakage := harm of always_answer (the policy that ignores risk)
        if "always_answer" in agg[ek]:
            leakage.append(agg[ek]["always_answer"]["harm_mean"])
        else:
            leakage.append(0.0)
    if not utility:
        return None
    # Sort by epsilon ascending so the X-axis reads small-to-large
    order = sorted(range(len(eps_values)), key=lambda i: eps_values[i])
    eps_values = [eps_values[i] for i in order]
    utility = [utility[i] for i in order]
    leakage = [leakage[i] for i in order]
    fig = plot_fn(eps_values=eps_values, utility=utility, leakage=leakage)
    target = out_dir / "r4_privacy_utility"
    save_fn(fig, str(target))
    return str(target)


def _make_r5_figure(r5_run: Path, out_dir: Path, *, plot_fn, save_fn) -> str | None:
    data = _load(r5_run / "r5.json")
    agg = data["aggregated"]
    if not agg:
        return None
    configs = [f"{a['backend'][:14]}/{a['config']}" for a in agg]
    # Collect all phase names across combos
    all_phases: set[str] = set()
    for a in agg:
        all_phases.update(a.get("phases", {}).keys())
    phase_names = sorted(all_phases)
    # phase_data[phase] = list of tokens-per-claim values, one per combo
    phase_data: dict[str, list[float]] = {}
    for ph in phase_names:
        phase_data[ph] = [
            a.get("phases", {}).get(ph, {}).get("tokens_mean", 0.0)
            for a in agg
        ]
    fig = plot_fn(configs=configs, phase_data=phase_data)
    target = out_dir / "r5_overhead"
    save_fn(fig, str(target))
    return str(target)


def _utility_per_k_from_r2(r2_run: Path | None, ks: list[int]) -> list[float]:
    """Compute mean(max-F1 over first k branches) per k from R2 per-seed data."""
    if r2_run is None:
        return [0.0] * len(ks)
    util_by_k: dict[int, list[float]] = {}
    for sf in sorted(r2_run.glob("seed_*.json")):
        sd = _load(sf)
        for ex in sd.get("per_example", []):
            branches = ex.get("branches", [])
            for k in ks:
                if not branches:
                    continue
                f1s = [b.get("f1", 0.0) for b in branches[:k]]
                if f1s:
                    util_by_k.setdefault(k, []).append(max(f1s))
    return [float(np.mean(util_by_k[k])) if util_by_k.get(k) else 0.0 for k in ks]


def _make_r5_vs_k_figure(
    r5_run: Path, results_root: Path, out_dir: Path, *, plot_fn, save_fn,
) -> str | None:
    data = _load(r5_run / "r5.json")
    agg = data["aggregated"]
    # Pick out the pcg_kN configs and pair (k, tokens-per-claim).
    pairs: list[tuple[int, float]] = []
    for a in agg:
        cfg_name = a["config"]
        if cfg_name.startswith("pcg_k"):
            try:
                k = int(cfg_name.split("k")[-1])
            except ValueError:
                continue
            pairs.append((k, a["tokens_per_claim_mean"]))
    if not pairs:
        return None
    pairs.sort(key=lambda p: p[0])
    ks = [p[0] for p in pairs]
    tokens_per_claim = [p[1] for p in pairs]
    # Utility per k pulled from R2 if available
    r2_run = _latest_run(results_root, "r2")
    utility = _utility_per_k_from_r2(r2_run, ks)
    fig = plot_fn(ks=ks, tokens_per_claim=tokens_per_claim, utility=utility)
    target = out_dir / "r5_overhead_vs_k"
    save_fn(fig, str(target))
    return str(target)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(results_dir: str = "results", out: str = "figures") -> int:
    results_root = Path(results_dir)
    if not results_root.is_absolute():
        results_root = project_root() / results_root
    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_section(f"make_figures: results={results_root}  out={out_dir}")

    from pcg.eval.plots import (
        plot_intro_hero,                     # noqa: F401  used by make_intro_hero
        plot_r1_audit_decomposition,
        plot_r2_redundancy_law,
        plot_r3_responsibility_heatmap,
        plot_r4_privacy_utility,
        plot_r4_risk_pareto,
        plot_r5_overhead,
        plot_r5_overhead_vs_k,
        save_fig,
        set_style,
    )
    set_style()

    artifacts: list[dict] = []
    handlers: list[tuple[str, callable]] = [
        ("r1", lambda r: _make_r1_figure(r, out_dir, plot_fn=plot_r1_audit_decomposition, save_fn=save_fig)),
        ("r2", lambda r: _make_r2_figure(r, out_dir, plot_fn=plot_r2_redundancy_law, save_fn=save_fig)),
        ("r3", lambda r: _make_r3_figure(r, out_dir, plot_fn=plot_r3_responsibility_heatmap, save_fn=save_fig)),
        ("r4", lambda r: _make_r4_pareto_figure(r, out_dir, plot_fn=plot_r4_risk_pareto, save_fn=save_fig)),
        ("r4", lambda r: _make_r4_privacy_figure(r, out_dir, plot_fn=plot_r4_privacy_utility, save_fn=save_fig)),
        ("r5", lambda r: _make_r5_figure(r, out_dir, plot_fn=plot_r5_overhead, save_fn=save_fig)),
        ("r5", lambda r: _make_r5_vs_k_figure(r, results_root, out_dir, plot_fn=plot_r5_overhead_vs_k, save_fn=save_fig)),
    ]
    for rid, handler in handlers:
        run = _latest_run(results_root, rid)
        if run is None:
            log_info(f"  skip {rid}: no run found")
            continue
        try:
            target = handler(run)
            if target:
                log_info(f"  wrote {Path(target).name}.{{pdf,png}}")
                artifacts.append({"source_run": str(run), "target": target})
        except Exception as e:
            log_info(f"  ERROR for {rid}: {type(e).__name__}: {e}")

    manifest = {"figures": artifacts, "out_dir": str(out_dir)}
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    log_info(f"Wrote {len(artifacts)} figures + manifest.json")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="figures")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(results_dir=a.results_dir, out=a.out))
