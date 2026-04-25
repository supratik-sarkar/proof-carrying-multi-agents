"""
scripts/make_intro_hero.py — uses plots_v2.plot_intro_hero_v2.

Reads the latest R1 / R2 / R4 runs and feeds them into the new hero figure:
schematic banner + Safety + Audit + Tradeoff. Auto-detects mock-backend
source runs and applies a "PREVIEW ONLY" watermark.

Output:
    figures/intro_hero.{pdf,png}
    figures/intro_hero_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from _common import log_info, log_section, project_root


def _latest_run(results_dir: Path, rid: str) -> Path | None:
    base = results_dir / rid
    if not base.exists():
        return None
    candidates = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for c in candidates:
        if (c / f"{rid}.json").exists():
            return c
    return None


def _load_json(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def _extract_safety(r2_run: Path, r1_run: Path | None) -> dict | None:
    """Build the data dict for the Safety panel from R2 (and R1 for baseline)."""
    if r2_run is None:
        return None
    r2 = _load_json(r2_run / "r2.json")
    per_k = r2.get("aggregated_per_k") or []
    if not per_k:
        return None

    ks = [int(row["k"]) for row in per_k]
    empirical = [float(row["empirical_mean"]) for row in per_k]
    empirical_ci = [tuple(row["empirical_ci"]) for row in per_k]
    theory_curve = [
        float(row.get("theory_ucb_max", row.get("theory_plug_in_mean", 0.0)))
        for row in per_k
    ]

    # Baseline: empirical Pr(accept & wrong) WITHOUT a certificate.
    # Best estimate: R1's lhs_accept_and_wrong on the unprotected examples.
    # If unavailable, fall back to a heuristic (k=1 empirical × 1.5).
    baseline = None
    if r1_run is not None:
        try:
            r1 = _load_json(r1_run / "r1.json")
            lhs_field = r1.get("aggregated", {}).get("lhs_accept_and_wrong", {})
            mean_val = lhs_field.get("mean")
            if mean_val is not None:
                baseline = float(mean_val)
        except Exception:
            pass
    if baseline is None and empirical:
        baseline = float(empirical[0]) * 1.5

    return {
        "ks": ks,
        "empirical": empirical,
        "empirical_ci": empirical_ci,
        "theory_curve": theory_curve,
        "baseline_no_cert": baseline,
    }


def _extract_audit(r1_run: Path) -> dict | None:
    """Build the data dict for the Audit panel from R1."""
    if r1_run is None:
        return None
    r1 = _load_json(r1_run / "r1.json")
    agg = r1.get("aggregated") or {}
    if not agg:
        return None

    def _read(key: str):
        v = agg.get(key, {}) or {}
        m = float(v.get("mean", 0.0))
        ci = v.get("ci") or [m, m]
        return (m, (float(ci[0]), float(ci[1])))

    return {
        "channels": {
            "int_fail":     _read("p_int_fail"),
            "replay_fail":  _read("p_replay_fail"),
            "check_fail":   _read("p_check_fail"),
            "cov_gap":      _read("p_cov_gap"),
        },
        "lhs": _read("lhs_accept_and_wrong"),
    }


def _extract_tradeoff(r4_run: Path) -> dict | None:
    """Build the data dict for the Tradeoff panel from R4."""
    if r4_run is None:
        return None
    r4 = _load_json(r4_run / "r4.json")
    agg = r4.get("aggregated") or {}
    if not agg:
        return None

    eps_keys = list(agg.keys())
    policies: dict[str, dict] = {}
    for pname in ("always_answer", "threshold_pcg", "learned"):
        costs: list[float] = []
        harms: list[float] = []
        for ek in eps_keys:
            row = agg.get(ek, {}).get(pname)
            if row is None:
                continue
            costs.append(float(row.get("cost_mean", 0.0)))
            harms.append(float(row.get("harm_mean", 0.0)))
        if costs:
            policies[pname] = {"costs": costs, "harms": harms}
    if not policies:
        return None
    return {"policies": policies}


def main(results_dir: str = "results", out: str = "figures") -> int:
    results_root = Path(results_dir)
    if not results_root.is_absolute():
        results_root = project_root() / results_root
    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_section(f"make_intro_hero (v2): results={results_root}  out={out_dir}")

    from pcg.eval.plots_v2 import (
        BOLD_THEME,
        detect_mock_runs,
        plot_intro_hero_v2,
        save_fig_v2,
    )

    r1_run = _latest_run(results_root, "r1")
    r2_run = _latest_run(results_root, "r2")
    r4_run = _latest_run(results_root, "r4")

    if r2_run: log_info(f"  R2 run: {r2_run.name}")
    if r1_run: log_info(f"  R1 run: {r1_run.name}")
    if r4_run: log_info(f"  R4 run: {r4_run.name}")

    safety   = _extract_safety(r2_run, r1_run)   if r2_run else None
    audit    = _extract_audit(r1_run)            if r1_run else None
    tradeoff = _extract_tradeoff(r4_run)         if r4_run else None

    is_mock = detect_mock_runs([r1_run, r2_run, r4_run])
    if is_mock:
        log_info("  Detected MOCK backend in source runs — watermark will be added")

    source_runs = [d.name for d in (r1_run, r2_run, r4_run) if d is not None]

    fig = plot_intro_hero_v2(
        safety=safety, audit=audit, tradeoff=tradeoff,
        theme=BOLD_THEME,
        source_runs=source_runs,
        is_mock=is_mock,
    )
    written = save_fig_v2(fig, str(out_dir / "intro_hero"))
    for w in written:
        log_info(f"  wrote {w}")

    manifest = {
        "name": "intro_hero_v2",
        "source_r1": str(r1_run) if r1_run else None,
        "source_r2": str(r2_run) if r2_run else None,
        "source_r4": str(r4_run) if r4_run else None,
        "is_mock": is_mock,
    }
    with (out_dir / "intro_hero_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="figures")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(results_dir=a.results_dir, out=a.out))
