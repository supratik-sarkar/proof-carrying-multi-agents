"""
Diverse coverage selection for the per-experiment plots (Phase M).

Each R-plot (R1..R5) shows 3 (LLM, dataset) cells out of the full
5-LLM × 8-dataset = 40-cell matrix. The selection criterion is
*representativeness*, not "top performers" — that distinction matters
because it stays honest under both smoke and real-LLM runs.

Diversity constraints:
    1. No LLM appears twice within the same R-plot
    2. No dataset appears twice within the same R-plot
    3. Across the 5 R-plots, every LLM appears at least 2-3 times
    4. Across the 5 R-plots, every dataset appears at least once
    5. Each (LLM, dataset) pair appears at most once across all plots
       so the figures don't show the same cell twice

The locally-runnable LLM cohort excludes Llama-3.3-70B and deepseek-v3
because the user's MacBook can't run them. The full 7-LLM cohort is
still produced when paired_pick is called with `include_large_llms=True`,
which is what `pick_top_k.py` uses post-Colab.

The resulting plan is JSON-serializable, deterministic given seed, and
can be overwritten by `pick_top_k.py` once real R-run JSONs exist.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------

LOCAL_LLMS = (
    "phi-3.5-mini",
    "qwen2.5-7B",
    "deepseek-llm-7b-chat",
    "Llama-3.1-8B",
    "Gemma-2-9b-it",
)

LARGE_LLMS = (
    "Llama-3.3-70B",
    "deepseek-v3",
)

ALL_DATASETS = (
    "hotpotqa",
    "twowiki",
    "toolbench",
    "fever",
    "pubmedqa",
    "tatqa",
    "weblinx",
    "synthetic",
)

R_EXPERIMENTS = ("r1", "r2", "r3", "r4", "r5")


# ---------------------------------------------------------------------------
# Plan structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (LLM, dataset) selection."""
    llm: str
    dataset: str

    def __str__(self) -> str:
        return f"({self.llm}, {self.dataset})"


@dataclass
class CoveragePlan:
    """The full plan: which 3 cells each R-plot uses."""
    per_experiment: dict[str, list[Cell]] = field(default_factory=dict)
    rationale: str = ""

    def cells_for(self, r_id: str) -> list[Cell]:
        return list(self.per_experiment.get(r_id, []))

    def to_dict(self) -> dict:
        return {
            "per_experiment": {
                r: [{"llm": c.llm, "dataset": c.dataset} for c in cells]
                for r, cells in self.per_experiment.items()
            },
            "rationale": self.rationale,
        }

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def read_json(cls, path: Path) -> "CoveragePlan":
        d = json.loads(Path(path).read_text())
        return cls(
            per_experiment={
                r: [Cell(c["llm"], c["dataset"]) for c in cells]
                for r, cells in d.get("per_experiment", {}).items()
            },
            rationale=d.get("rationale", ""),
        )

    def coverage_summary(self) -> dict:
        """Audit how well the plan covers the LLM and dataset axes."""
        llm_counts: dict[str, int] = {}
        dataset_counts: dict[str, int] = {}
        cells_used: set[tuple[str, str]] = set()
        for r, cells in self.per_experiment.items():
            for c in cells:
                llm_counts[c.llm] = llm_counts.get(c.llm, 0) + 1
                dataset_counts[c.dataset] = dataset_counts.get(c.dataset, 0) + 1
                cells_used.add((c.llm, c.dataset))
        return {
            "n_total_cells_chosen": sum(
                len(cs) for cs in self.per_experiment.values()),
            "n_unique_cells": len(cells_used),
            "llm_coverage": dict(sorted(llm_counts.items())),
            "dataset_coverage": dict(sorted(dataset_counts.items())),
        }


# ---------------------------------------------------------------------------
# Diverse-coverage builder
# ---------------------------------------------------------------------------


def build_diverse_coverage(
    *,
    llms: Sequence[str] = LOCAL_LLMS,
    datasets: Sequence[str] = ALL_DATASETS,
    experiments: Sequence[str] = R_EXPERIMENTS,
    cells_per_experiment: int = 3,
) -> CoveragePlan:
    """Produce a plan where each R-plot has 3 cells, with no LLM-or-
    dataset repeats within an R-plot, and good spread across the matrix.

    Algorithm: assign cells in a round-robin pattern. Each R-plot
    consumes a different "slice" of the (llm × dataset) cartesian
    product so no two plots share a cell. With 5 LLMs × 8 datasets
    = 40 cells available and 5 × 3 = 15 cells consumed, we leave
    25 unused — plenty of slack to enforce diversity.
    """
    n_llms = len(llms)
    n_data = len(datasets)
    n_exp = len(experiments)
    if n_exp * cells_per_experiment > n_llms * n_data:
        raise ValueError(
            f"asked for {n_exp * cells_per_experiment} cells but matrix "
            f"only has {n_llms * n_data}"
        )

    plan: dict[str, list[Cell]] = {r: [] for r in experiments}

    # Round-robin LLM assignment per experiment, with a per-experiment
    # offset so the LLM cycle starts at a different place. This spreads
    # each LLM across multiple experiments.
    used_cells: set[tuple[str, str]] = set()
    used_llms_per_exp: dict[str, set[str]] = {r: set() for r in experiments}
    used_data_per_exp: dict[str, set[str]] = {r: set() for r in experiments}

    for ei, exp in enumerate(experiments):
        for ci in range(cells_per_experiment):
            llm_idx = (ei + ci * 2) % n_llms          # stride 2 to vary
            data_idx = (ei * 3 + ci * 5) % n_data     # stride 5 to vary
            # Find a free (LLM, dataset) cell respecting all constraints
            llm = llms[llm_idx]
            ds  = datasets[data_idx]
            tries = 0
            while (
                (llm, ds) in used_cells
                or llm in used_llms_per_exp[exp]
                or ds in used_data_per_exp[exp]
            ) and tries < n_llms * n_data:
                # rotate
                llm_idx = (llm_idx + 1) % n_llms
                if llm_idx == (ei + ci * 2) % n_llms:
                    data_idx = (data_idx + 1) % n_data
                llm = llms[llm_idx]
                ds  = datasets[data_idx]
                tries += 1
            if (llm, ds) in used_cells:
                # extremely unlikely with our matrix sizes; skip cell
                continue
            plan[exp].append(Cell(llm=llm, dataset=ds))
            used_cells.add((llm, ds))
            used_llms_per_exp[exp].add(llm)
            used_data_per_exp[exp].add(ds)

    rationale = (
        f"Diverse coverage plan over {n_llms} LLMs × {n_data} datasets. "
        f"Each of {n_exp} experiments shows {cells_per_experiment} cells, "
        "with no within-experiment LLM or dataset repetition. "
        f"Locally-runnable LLM cohort: {', '.join(llms)}. "
        "Replace via scripts/pick_top_k.py once real R-run JSONs exist."
    )
    return CoveragePlan(per_experiment=plan, rationale=rationale)


# ---------------------------------------------------------------------------
# Default plan path used by the R-plot helpers
# ---------------------------------------------------------------------------


DEFAULT_PLAN_PATH = Path("artifacts/coverage_plan.json")


def load_or_build_plan(
    path: Path | str = DEFAULT_PLAN_PATH,
    *, force_rebuild: bool = False,
) -> CoveragePlan:
    """Convenience: load existing plan, otherwise build the default
    diverse-coverage plan and persist it."""
    p = Path(path)
    if p.exists() and not force_rebuild:
        return CoveragePlan.read_json(p)
    plan = build_diverse_coverage()
    plan.write_json(p)
    return plan
