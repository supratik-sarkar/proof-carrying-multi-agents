from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_TABLES = RESULTS / "tables"
RESULTS_TABLES_CSV = RESULTS_TABLES / "csv"
RESULTS_TABLES_TEX = RESULTS_TABLES / "tex"

LATEX = ROOT / "latex"
LATEX_EXPERIMENTS = LATEX / "experiments.tex"
LATEX_APPENDIX_EXP_DETAILS = LATEX / "appendix_exp_details.tex"

WORKFLOW = ROOT / "workflow"
NOTEBOOKS = ROOT / "notebooks"
ARTIFACTS = ROOT / "artifacts"


def ensure_output_dirs() -> None:
    for path in [
        RESULTS_FIGURES,
        RESULTS_TABLES_CSV,
        RESULTS_TABLES_TEX,
        WORKFLOW,
        NOTEBOOKS,
        ARTIFACTS,
    ]:
        path.mkdir(parents=True, exist_ok=True)
