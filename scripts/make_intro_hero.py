"""
scripts/make_intro_hero.py

Builds the page-1 hero figure for the Introduction.

Two panels:
    Left:  log-scale false-accept rate vs k, with no-cert baseline.
    Right: utility (mean F1) vs k, with no-cert baseline.

Reads from:
    results/r2/<latest>/r2.json     - false-accept rate vs k
    results/r2/<latest>/seed_*.json - per-example branch F1 scores
    results/r1/<latest>/r1.json     - no-cert utility baseline (optional)

Output:
    figures/intro_hero.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def _load(p: Path):
    with p.open("r") as fh:
        return json.load(fh)


def _utility_per_k_from_r2(r2_run: Path, ks: list[int]) -> list[float]:
    """Mean over examples of max-F1 across the first k branches."""
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


def _no_cert_utility_baseline(r1_run: Path | None) -> float | None:
    """Mean F1 across non-attacked R1 examples (the "no certificate" baseline)."""
    if r1_run is None:
        return None
    f1s = []
    for sf in sorted(r1_run.glob("seed_*.json")):
        sd = _load(sf)
        for r in sd.get("per_example", []):
            if r.get("attack") in (None, "none") and "f1_to_gold" in r:
                f1s.append(r["f1_to_gold"])
    return float(np.mean(f1s)) if f1s else None


def _no_cert_false_accept_baseline(r1_run: Path | None) -> float | None:
    """Empirical Pr(accept & wrong) from R1 — the LHS aggregate."""
    if r1_run is None:
        return None
    data = _load(r1_run / "r1.json")
    agg = data.get("aggregated", {})
    if "lhs_accept_and_wrong" in agg:
        return float(agg["lhs_accept_and_wrong"]["mean"])
    return None


def main(results_dir: str = "results", out: str = "figures") -> int:
    results_root = Path(results_dir)
    if not results_root.is_absolute():
        results_root = project_root() / results_root
    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_section(f"make_intro_hero: results={results_root}  out={out_dir}")

    from pcg.eval.plots import plot_intro_hero, save_fig, set_style
    set_style()

    r2_run = _latest_run(results_root, "r2")
    r1_run = _latest_run(results_root, "r1")

    if r2_run is None:
        log_info("  no R2 run found - cannot make hero figure")
        return 1
    log_info(f"  R2 run: {r2_run.name}")
    if r1_run:
        log_info(f"  R1 run: {r1_run.name}")

    # --- false-accept rate vs k from R2 ---
    r2 = _load(r2_run / "r2.json")
    per_k = r2.get("aggregated_per_k", [])
    if not per_k:
        log_info("  R2 has no aggregated_per_k - skipping")
        return 1
    ks = [int(row["k"]) for row in per_k]
    fa_with_cert = [float(row["empirical_mean"]) for row in per_k]

    # --- utility (max-F1 over k branches) per k ---
    util_with_cert = _utility_per_k_from_r2(r2_run, ks)

    # --- baselines ---
    fa_baseline = _no_cert_false_accept_baseline(r1_run)
    if fa_baseline is None:
        # Fallback: use the k=1 empirical false-accept (single branch, like no-cert).
        fa_baseline = fa_with_cert[0] if fa_with_cert else 0.0
    util_baseline = _no_cert_utility_baseline(r1_run)
    if util_baseline is None:
        util_baseline = util_with_cert[0] if util_with_cert else 0.0

    fig = plot_intro_hero(
        ks=ks,
        utility_without_cert=util_baseline,
        utility_with_cert=util_with_cert,
        false_accept_without_cert=fa_baseline,
        false_accept_with_cert=fa_with_cert,
    )
    target = out_dir / "intro_hero"
    save_fig(fig, str(target))
    log_info(f"  wrote {target.name}.{{pdf,png}}")

    manifest_path = out_dir / "intro_hero_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump({
            "name": "intro_hero",
            "source_r2": str(r2_run / "r2.json"),
            "source_r1": str(r1_run / "r1.json") if r1_run else None,
            "ks": ks,
            "fa_with_cert": fa_with_cert,
            "util_with_cert": util_with_cert,
            "fa_baseline": fa_baseline,
            "util_baseline": util_baseline,
        }, fh, indent=2)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="figures")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(results_dir=a.results_dir, out=a.out))
