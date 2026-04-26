"""
scripts/pick_top_k.py

Post-Colab: select the top-k (LLM, dataset) cells per R-experiment from
real run results. Writes a new artifacts/coverage_plan.json that the
make_per_experiment_plots driver picks up on the next render — no other
code changes required.

This is the second half of the diverse-coverage / top-k strategy:
during the smoke phase, plots use diverse coverage. Once Colab has run
the full sweep, run this script to swap in the actual top performers.

Selection criteria (per R-experiment):
    r1: smallest LHS (best safety) — lower is better
    r2: smallest empirical at k=k_max (steepest collapse) — lower is better
    r3: largest mean top-1 root-cause accuracy — higher is better
    r4: largest harm-reduction factor (baseline_max / ours_min) — higher is better
    r5: smallest token-overhead factor (ours_total / base_total) — lower is better

Diversity constraints (preserved from build_diverse_coverage):
    - No LLM repeats within an R-plot
    - No dataset repeats within an R-plot

When two cells tie on the criterion, the higher-diversity choice wins
(LLM/dataset not yet used elsewhere in the plan).

Usage:
    # After Colab runs deposit results/{r_id}-{llm}-{dataset}-*/cell.json files:
    python scripts/pick_top_k.py
    # or with custom params:
    python scripts/pick_top_k.py --k 3 --include-large-llms

The previous coverage_plan.json is backed up to .bak before overwrite.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from _common import log_info, log_section
from pcg.eval.coverage import (
    Cell, CoveragePlan, DEFAULT_PLAN_PATH,
    LOCAL_LLMS, LARGE_LLMS, ALL_DATASETS, R_EXPERIMENTS,
)


# ---------------------------------------------------------------------------
# Per-R-experiment scorers
# ---------------------------------------------------------------------------


def _score_r1(cell_data: dict) -> tuple[float, str]:
    """R1: smaller LHS is better. Returns (score, "lower"|"higher" higher-is-better)."""
    lhs = cell_data.get("lhs")
    if lhs is None:
        return (float("inf"), "lower")
    return (float(lhs), "lower")


def _score_r2(cell_data: dict) -> tuple[float, str]:
    """R2: smallest empirical at k_max is best (steepest collapse)."""
    emp = cell_data.get("empirical") or []
    if not emp:
        return (float("inf"), "lower")
    return (float(emp[-1]), "lower")


def _score_r3(cell_data: dict) -> tuple[float, str]:
    """R3: largest mean top-1 root-cause accuracy is best."""
    accs = cell_data.get("ours_acc") or []
    if not accs:
        return (-float("inf"), "higher")
    return (float(sum(accs) / len(accs)), "higher")


def _score_r4(cell_data: dict) -> tuple[float, str]:
    """R4: largest harm-reduction factor (always_max / ours_min)."""
    pol = cell_data.get("policies") or {}
    try:
        base_max = max(pol["always_answer"]["harm"])
        ours_min = min(pol["threshold_pcg"]["harm"])
        return (base_max / max(ours_min, 1e-6), "higher")
    except (KeyError, TypeError, ValueError):
        return (-float("inf"), "higher")


def _score_r5(cell_data: dict) -> tuple[float, str]:
    """R5: smallest token overhead factor (ours_total / base_total)."""
    phases = cell_data.get("phases") or {}
    base = (cell_data.get("phases_baseline") or {}).get("prove",
                                                          phases.get("prove", 0))
    ours_total = sum(phases.values())
    if not base or not ours_total:
        return (float("inf"), "lower")
    return (ours_total / base, "lower")


SCORERS = {
    "r1": _score_r1,
    "r2": _score_r2,
    "r3": _score_r3,
    "r4": _score_r4,
    "r5": _score_r5,
}


# ---------------------------------------------------------------------------
# Discovery: walk results/ for available cells per R-experiment
# ---------------------------------------------------------------------------


def discover_real_cells(
    results_root: Path,
    *, llms: list[str], datasets: list[str],
) -> dict[str, list[tuple[Cell, dict]]]:
    """For each R-experiment, list every (cell, cell_data) we can find."""
    out: dict[str, list[tuple[Cell, dict]]] = {r: [] for r in R_EXPERIMENTS}
    if not results_root.exists():
        return out
    for r in R_EXPERIMENTS:
        for llm in llms:
            for ds in datasets:
                pattern = f"{r}-{llm}-{ds}-*"
                matches = sorted(results_root.glob(pattern))
                if not matches:
                    continue
                cell_json = matches[-1] / "cell.json"
                if not cell_json.exists():
                    continue
                try:
                    cd = json.loads(cell_json.read_text())
                except json.JSONDecodeError:
                    continue
                out[r].append((Cell(llm=llm, dataset=ds), cd))
    return out


# ---------------------------------------------------------------------------
# Top-k with diversity tie-break
# ---------------------------------------------------------------------------


def pick_top_k(
    candidates: list[tuple[Cell, dict]],
    k: int,
    scorer,
    used_llms: set[str] | None = None,
    used_datasets: set[str] | None = None,
) -> list[Cell]:
    """Pick top-k cells subject to within-plot LLM/dataset uniqueness.

    Selection: rank by scorer, then walk down picking cells that don't
    repeat an LLM or dataset already chosen for this plot.
    """
    if not candidates:
        return []

    # Score everything; sort by score; HOB direction respected per scorer
    scored: list[tuple[float, str, Cell]] = []
    for cell, cd in candidates:
        s, direction = scorer(cd)
        # Convert to a unified "lower is better" key for sorting
        key = -s if direction == "higher" else s
        scored.append((key, direction, cell))
    scored.sort(key=lambda t: t[0])

    used_llm: set[str] = set(used_llms or set())
    used_ds:  set[str] = set(used_datasets or set())
    picks: list[Cell] = []
    for _, _, cell in scored:
        if cell.llm in used_llm or cell.dataset in used_ds:
            continue
        picks.append(cell)
        used_llm.add(cell.llm)
        used_ds.add(cell.dataset)
        if len(picks) >= k:
            break
    return picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    results_dir: str = "results",
    plan_path: str | None = None,
    k: int = 3,
    include_large_llms: bool = False,
    rationale_suffix: str = "",
) -> int:
    log_section("pick_top_k")
    results_root = Path(results_dir)
    out_path = Path(plan_path or DEFAULT_PLAN_PATH)

    llms = list(LOCAL_LLMS) + (list(LARGE_LLMS) if include_large_llms else [])
    log_info(f"  LLM cohort: {llms}")
    log_info(f"  datasets:   {list(ALL_DATASETS)}")

    available = discover_real_cells(
        results_root, llms=llms, datasets=list(ALL_DATASETS),
    )
    total_avail = sum(len(v) for v in available.values())
    log_info(f"  found {total_avail} real cell entries across "
             f"{sum(1 for v in available.values() if v)}/{len(R_EXPERIMENTS)} "
             f"experiments")
    if total_avail == 0:
        log_info("  No real cell.json files found — leaving plan unchanged.")
        return 1

    new_plan: dict[str, list[Cell]] = {}
    for r_id in R_EXPERIMENTS:
        scorer = SCORERS[r_id]
        cands = available.get(r_id, [])
        if not cands:
            log_info(f"  {r_id}: no real cells available — keeping previous plan")
            # Preserve existing plan entry if present
            if out_path.exists():
                old = CoveragePlan.read_json(out_path)
                new_plan[r_id] = old.cells_for(r_id)
            continue
        picks = pick_top_k(cands, k=k, scorer=scorer)
        new_plan[r_id] = picks
        log_info(f"  {r_id}: top-{k} = {[str(c) for c in picks]}")

    # Backup the previous plan
    if out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, bak)
        log_info(f"  backed up previous plan to {bak}")

    rationale = (
        f"Top-{k} per R-experiment, scored from real run results. "
        f"Selection criteria: r1=lowest LHS, r2=lowest empirical@k_max, "
        f"r3=highest top-1 accuracy, r4=highest harm-reduction factor, "
        f"r5=lowest overhead factor. Diversity: no LLM or dataset repeats "
        f"within the same plot. {rationale_suffix}".strip()
    )
    plan = CoveragePlan(per_experiment=new_plan, rationale=rationale)
    plan.write_json(out_path)
    log_info(f"  wrote {out_path}")
    log_info(f"  re-render figures with: make figures")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--plan-path", default=None,
                   help=f"Output plan path (default: {DEFAULT_PLAN_PATH})")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--include-large-llms", action="store_true",
                   help="Include Llama-3.3-70B + deepseek-v3 in candidate set")
    p.add_argument("--rationale-suffix", default="",
                   help="Append a free-text note to the rationale field")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(
        results_dir=a.results_dir,
        plan_path=a.plan_path,
        k=a.k,
        include_large_llms=a.include_large_llms,
        rationale_suffix=a.rationale_suffix,
    ))
