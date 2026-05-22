#!/usr/bin/env bash

# =====================================================================
# Merge AgentRR artifacts into PCG-MAS figures/tables
# =====================================================================

echo "Merge AgentRR artifacts into PCG-MAS figures/tables"

PCG_ROOT="$(pwd)"

printf "PCG-MAS virtual environment name [default: multi-agents]: "
read -r PCG_VENV_NAME
PCG_VENV_NAME="${PCG_VENV_NAME:-multi-agents}"

PCG_VENV="$PCG_ROOT/$PCG_VENV_NAME"

if [[ ! -d "$PCG_VENV" ]]; then
  echo "PCG-MAS venv not found: $PCG_VENV"
  exit 1
fi

deactivate 2>/dev/null || true
source "$PCG_VENV/bin/activate"
export PYTHONPATH="$PCG_ROOT:${PYTHONPATH:-}"

AGENTRR_ROOT="results/baselines/agentrr/r1_r5"
AGENTRR_AGG="$AGENTRR_ROOT/aggregate_by_dataset_model.json"
AGENTRR_MANIFEST="$AGENTRR_ROOT/manifest.json"

if [[ ! -f "$AGENTRR_AGG" ]]; then
  echo "Missing AgentRR aggregate file: $AGENTRR_AGG"
  echo "Run AgentRR first:"
  echo "  bash scripts/baselines/agentrr/run_agentrr_interactive.sh"
  exit 1
fi

if [[ ! -f "$AGENTRR_MANIFEST" ]]; then
  echo "Missing AgentRR manifest file: $AGENTRR_MANIFEST"
  exit 1
fi

echo "Found AgentRR artifacts:"
echo "  $AGENTRR_AGG"
echo "  $AGENTRR_MANIFEST"

python -m py_compile scripts/figures/make_paper_figures.py
python -m py_compile scripts/tables/collect_paper_metrics.py
python -m py_compile scripts/tables/repair_paper_metrics_metadata.py 2>/dev/null || true
python -m py_compile scripts/tables/make_paper_tables.py

python scripts/tables/collect_paper_metrics.py
if [[ -f scripts/tables/repair_paper_metrics_metadata.py ]]; then
  python scripts/tables/repair_paper_metrics_metadata.py
fi

# First overlay ShieldAgent if present, using existing runner mechanics,
# then overlay AgentRR. The ShieldAgent runner rebuilds once; AgentRR rebuilds final.
if [[ -f results/baselines/shieldagent/r1_r5/aggregate_by_dataset_model.json && -x scripts/baselines/shieldagent/merge_shieldagent_into_pcgmas.sh ]]; then
  echo ""
  echo "ShieldAgent artifacts detected; running ShieldAgent merge first."
  printf "\n" | bash scripts/baselines/shieldagent/merge_shieldagent_into_pcgmas.sh
fi

echo ""
echo "Overlaying real AgentRR artifacts into paper_metrics.jsonl."

python scripts/baselines/agentrr/overlay_agentrr_into_pcgmas.py

python scripts/tables/validate_paper_metrics.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --allow-partial

rm -rf results/figures_staged_selected results/tables/tex_staged_selected
mkdir -p results/figures_staged_selected results/tables/tex_staged_selected results/figures results/tables/tex results/audit

python scripts/figures/make_paper_figures.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --outdir results/figures_staged_selected \
  --allow-partial

python scripts/tables/make_paper_tables.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --outdir results/tables/tex_staged_selected \
  --allow-partial

python - <<'PY'
import json
import shutil
from pathlib import Path

staged_fig = Path("results/figures_staged_selected")
final_fig = Path("results/figures")
staged_tab = Path("results/tables/tex_staged_selected")
final_tab = Path("results/tables/tex")

# AgentRR is an appendix/hero overlay for this workflow. It must not
# prune or overwrite unrelated main/optional figures. Copy only the two
# hero figures that can visibly include AgentRR; leave all other files in
# results/figures and results/tables/tex untouched.
figures = [
    "intro_hero_v4",
    "appendix_hero_v4",
]

tables = [
]

copied_figures, missing_figures = [], []
for stem in figures:
    found = False
    for ext in [".pdf", ".png"]:
        src = staged_fig / f"{stem}{ext}"
        if src.exists():
            dst = final_fig / src.name
            shutil.copy2(src, dst)
            copied_figures.append(str(dst))
            found = True
    if not found:
        missing_figures.append(stem)

copied_tables, missing_tables = [], []
for stem in tables:
    src = staged_tab / f"{stem}.tex"
    if src.exists():
        dst = final_tab / src.name
        shutil.copy2(src, dst)
        copied_tables.append(str(dst))
    else:
        missing_tables.append(stem)

manifest = {
    "copied_figures": copied_figures,
    "missing_figures": missing_figures,
    "copied_tables": copied_tables,
    "missing_tables": missing_tables,
    "note": "Built after overlaying AgentRR artifacts. Only intro_hero_v4 and appendix_hero_v4 are copied so existing main/optional figures and tables are preserved.",
}
Path("results/audit/agentrr_merge_build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo ""
echo "Final figures:"
find results/figures -maxdepth 1 -type f | sort

echo ""
echo "Final LaTeX tables:"
find results/tables/tex -maxdepth 1 -type f | sort

echo "AGENTRR_MERGED_INTO_PCGMAS_COMPLETE"
