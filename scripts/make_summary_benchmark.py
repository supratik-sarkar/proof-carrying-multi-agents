"""
scripts/make_summary_benchmark.py — comprehensive multi-backend overview.

Aggregates results across backends from multiple R-runs into a single
2x2 panel suitable for the appendix or project website.

Output:
    figures/summary_benchmark.{pdf,png}
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


def _short_backend_label(s: str) -> str:
    """e.g. 'meta-llama/Llama-3.3-70B-Instruct' -> 'Llama-3.3-70B'."""
    name = str(s).split("/")[-1]
    name = name.replace("-Instruct", "").replace("-instruct", "")
    return name[:18]


def _safety_per_backend(r1_run: Path | None) -> dict | None:
    """Single-backend R1 → 1-bar chart. Multi-backend support added later."""
    if r1_run is None:
        return None
    try:
        r1 = _load_json(r1_run / "r1.json")
    except Exception:
        return None

    cfg_snap_path = r1_run / "config_snapshot.json"
    backend_label = "unknown"
    if cfg_snap_path.exists():
        try:
            cfg = _load_json(cfg_snap_path)
            b = cfg.get("backend", {}) or {}
            backend_label = _short_backend_label(
                b.get("model_name") or b.get("kind") or "unknown"
            )
        except Exception:
            pass

    agg = r1.get("aggregated", {}) or {}
    lhs_field = agg.get("lhs_accept_and_wrong", {}) or {}
    mean_val = float(lhs_field.get("mean", 0.0))
    ci = lhs_field.get("ci") or [mean_val, mean_val]

    return {
        "backends": [backend_label],
        "values": [mean_val],
        "cis": [(float(ci[0]), float(ci[1]))],
        "metric_name": "False-accept rate",
        "title": "Safety per backend (R1)",
    }


def _overhead_per_backend(r5_run: Path | None) -> dict | None:
    if r5_run is None:
        return None
    try:
        r5 = _load_json(r5_run / "r5.json")
    except Exception:
        return None
    agg = r5.get("aggregated") or []
    if not agg:
        return None

    # Stable per-backend ordering
    backends: list[str] = []
    for entry in agg:
        b = _short_backend_label(entry.get("backend", "unknown"))
        if b not in backends:
            backends.append(b)

    # Pick a representative config per backend (prefer pcg_k1 then anything)
    chosen_per_backend: dict[str, dict] = {}
    for b in backends:
        candidates = [
            e for e in agg if _short_backend_label(e.get("backend", "")) == b
        ]
        if not candidates:
            continue
        preferred = next(
            (c for c in candidates if c.get("config") == "pcg_k1"), None
        ) or next(
            (c for c in candidates if c.get("config") == "baseline_no_pcg"), None
        ) or candidates[0]
        chosen_per_backend[b] = preferred

    phase_names: list[str] = []
    for entry in chosen_per_backend.values():
        for ph in (entry.get("phases") or {}):
            if ph not in phase_names:
                phase_names.append(ph)
    phase_names.sort()

    phases: dict[str, list[float]] = {}
    for ph in phase_names:
        vals: list[float] = []
        for b in backends:
            entry = chosen_per_backend.get(b)
            if entry is None:
                vals.append(0.0)
                continue
            ph_data = (entry.get("phases") or {}).get(ph, {}) or {}
            vals.append(float(ph_data.get("tokens_mean", 0.0)))
        phases[ph] = vals

    return {"backends": backends, "phases": phases}


def _responsibility_per_regime(r3_run: Path | None) -> dict | None:
    if r3_run is None:
        return None
    try:
        r3 = _load_json(r3_run / "r3.json")
    except Exception:
        return None
    agg = r3.get("aggregated") or {}
    if not agg:
        return None

    regimes = list(agg.keys())
    values: list[float] = []
    cis: list[tuple[float, float]] = []
    for r in regimes:
        v = agg.get(r, {}) or {}
        m = v.get("top1_accuracy_mean")
        m = float(m) if m is not None else 0.0
        ci = v.get("ci") or [m, m]
        values.append(m)
        cis.append((float(ci[0]), float(ci[1])))

    return {
        "backends": regimes,
        "values": values,
        "cis": cis,
        "metric_name": "Top-1 root-cause accuracy",
        "title": "Responsibility per regime (R3)",
    }


def _redundancy(r2_run: Path | None) -> dict | None:
    if r2_run is None:
        return None
    try:
        r2 = _load_json(r2_run / "r2.json")
    except Exception:
        return None
    per_k = r2.get("aggregated_per_k") or []
    if not per_k:
        return None
    return {
        "ks": [int(row["k"]) for row in per_k],
        "empirical": [float(row["empirical_mean"]) for row in per_k],
        "empirical_ci": [tuple(row["empirical_ci"]) for row in per_k],
        "theory_curve": [
            float(row.get("theory_ucb_max", 0.0)) for row in per_k
        ],
        "baseline_no_cert": None,
    }


def _tradeoff_from_r4(r4_run: Path | None) -> dict | None:
    """Build the Pareto-scatter data dict for R4 (cost vs harm across eps)."""
    if r4_run is None:
        return None
    try:
        r4 = _load_json(r4_run / "r4.json")
    except Exception:
        return None
    agg = r4.get("aggregated") or {}
    if not agg:
        return None

    eps_keys = list(agg.keys())
    policies: dict[str, dict] = {}
    for pname in ("always_answer", "threshold_pcg", "learned"):
        costs: list[float] = []
        harms: list[float] = []
        for ek in eps_keys:
            row = (agg.get(ek) or {}).get(pname)
            if row is None:
                continue
            costs.append(float(row.get("cost_mean", 0.0)))
            harms.append(float(row.get("harm_mean", 0.0)))
        if costs:
            policies[pname] = {"costs": costs, "harms": harms}
    if not policies:
        return None
    return {"policies": policies}


def _headline_numbers(
    r1_run: Path | None,
    r2_run: Path | None,
    r3_run: Path | None,
    r4_run: Path | None,
    r5_run: Path | None,
) -> list:
    """Compute 3-4 high-impact KPIs from the available R-runs.

    Each KPI is a {value, label, sub} dict. Missing source = KPI omitted
    (defensive; never crashes). Returns at most 4 KPIs, ordered by
    "impact"-first so the most striking number renders at the top.
    """
    kpis: list[dict] = []

    # --- 1. R2 safety: "{N}× fewer false accepts at k_max"
    if r2_run is not None:
        try:
            r2 = _load_json(r2_run / "r2.json")
            per_k = r2.get("aggregated_per_k") or []
            if per_k:
                emp_kmax = float(per_k[-1]["empirical_mean"])
                k_max = int(per_k[-1]["k"])
                # Baseline = R1's lhs_accept_and_wrong, fallback to k=1 empirical
                baseline = None
                if r1_run is not None:
                    try:
                        r1 = _load_json(r1_run / "r1.json")
                        v = (r1.get("aggregated", {}) or {}).get(
                            "lhs_accept_and_wrong", {}
                        )
                        if v.get("mean") is not None:
                            baseline = float(v["mean"])
                    except Exception:
                        pass
                if baseline is None:
                    baseline = float(per_k[0]["empirical_mean"]) * 1.5
                if emp_kmax > 0 and baseline > 0:
                    factor = baseline / emp_kmax
                    kpis.append({
                        "value": f"{factor:.0f}×",
                        "label": "fewer false accepts",
                        "sub": f"at k={k_max} vs no certificate",
                    })
        except Exception:
            pass

    # --- 2. R1 audit tightness: "{P}% LHS / RHS ratio"
    if r1_run is not None:
        try:
            r1 = _load_json(r1_run / "r1.json")
            agg = r1.get("aggregated", {}) or {}
            lhs = float(((agg.get("lhs_accept_and_wrong") or {}).get("mean")) or 0.0)
            rhs = sum(
                float(((agg.get(k) or {}).get("mean")) or 0.0)
                for k in (
                    "p_int_fail", "p_replay_fail",
                    "p_check_fail", "p_cov_gap",
                )
            )
            if rhs > 0:
                tightness = max(0.0, min(1.0, lhs / rhs))
                kpis.append({
                    "value": f"{tightness * 100:.0f}%",
                    "label": "Thm 1 bound tightness",
                    "sub": "LHS / RHS, lower = looser",
                })
        except Exception:
            pass

    # --- 3. R3 responsibility: top-1 root-cause accuracy (mean over regimes)
    if r3_run is not None:
        try:
            r3 = _load_json(r3_run / "r3.json")
            agg = r3.get("aggregated", {}) or {}
            vals: list[float] = []
            for v in agg.values():
                m = (v or {}).get("top1_accuracy_mean")
                if m is not None:
                    vals.append(float(m))
            if vals:
                mean_acc = sum(vals) / len(vals)
                kpis.append({
                    "value": f"{mean_acc * 100:.0f}%",
                    "label": "top-1 root-cause",
                    "sub": "accuracy averaged over regimes",
                })
        except Exception:
            pass

    # --- 4. R4 harm reduction: "{N}× lower harm vs always-answer"
    if r4_run is not None:
        try:
            tdata = _tradeoff_from_r4(r4_run)
            if tdata and "policies" in tdata:
                p = tdata["policies"]
                if "threshold_pcg" in p and "always_answer" in p:
                    our_min = min(p["threshold_pcg"]["harms"])
                    base_max = max(p["always_answer"]["harms"])
                    if our_min > 0 and base_max > 0:
                        ratio = base_max / our_min
                        kpis.append({
                            "value": f"{ratio:.0f}×",
                            "label": "lower harm",
                            "sub": "vs always-answer baseline",
                        })
        except Exception:
            pass

    # If nothing computable yet, emit a single placeholder KPI
    if not kpis:
        kpis = [{
            "value": "—",
            "label": "run experiments",
            "sub": "to populate KPIs",
        }]
    return kpis


def main(results_dir: str = "results", out: str = "figures") -> int:
    results_root = Path(results_dir)
    if not results_root.is_absolute():
        results_root = project_root() / results_root
    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_section(f"make_summary_benchmark: results={results_root}  out={out_dir}")

    from pcg.eval.plots_v2 import (
        BOLD_THEME,
        detect_mock_runs,
        plot_summary_benchmark,
        save_fig_v2,
    )

    r1 = _latest_run(results_root, "r1")
    r2 = _latest_run(results_root, "r2")
    r3 = _latest_run(results_root, "r3")
    r4 = _latest_run(results_root, "r4")
    r5 = _latest_run(results_root, "r5")

    is_mock = detect_mock_runs([r1, r2, r3, r4, r5])
    if is_mock:
        log_info("  Detected MOCK backend — watermark will be added")
    source_runs = [d.name for d in (r1, r2, r3, r4, r5) if d is not None]

    fig = plot_summary_benchmark(
        backend_safety=_safety_per_backend(r1),
        redundancy=_redundancy(r2),
        backend_responsibility=_responsibility_per_regime(r3),
        tradeoff=_tradeoff_from_r4(r4),
        backend_overhead=_overhead_per_backend(r5),
        headline_numbers=_headline_numbers(r1, r2, r3, r4, r5),
        theme=BOLD_THEME,
        source_runs=source_runs,
        is_mock=is_mock,
    )
    written = save_fig_v2(fig, str(out_dir / "summary_benchmark"))
    for w in written:
        log_info(f"  wrote {w}")

    manifest = {
        "name": "summary_benchmark",
        "sources": [str(d) for d in (r1, r2, r3, r4, r5) if d is not None],
        "is_mock": is_mock,
    }
    with (out_dir / "summary_benchmark_manifest.json").open("w") as f:
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
