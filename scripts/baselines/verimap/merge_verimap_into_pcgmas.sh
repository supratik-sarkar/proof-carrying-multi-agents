#!/usr/bin/env bash
set -euo pipefail

echo "Merge VeriMAP artifacts into PCG-MAS figures/tables"

PCG_ROOT="$(pwd)"

printf "PCG-MAS virtual environment name [default: multi-agents]: "
read -r PCG_VENV_NAME
PCG_VENV_NAME="${PCG_VENV_NAME:-multi-agents}"

PCG_VENV="$PCG_ROOT/$PCG_VENV_NAME"

if [[ -d "$PCG_VENV" ]]; then
  deactivate 2>/dev/null || true
  source "$PCG_VENV/bin/activate"
fi

export PYTHONPATH="$PCG_ROOT:${PYTHONPATH:-}"

if [[ ! -f results/baselines/verimap/r1_r5/aggregate_by_dataset_model.json ]]; then
  echo "Missing VeriMAP aggregate:"
  echo "  results/baselines/verimap/r1_r5/aggregate_by_dataset_model.json"
  exit 1
fi

if [[ ! -f results/baselines/verimap/r1_r5/manifest.json ]]; then
  echo "Missing VeriMAP manifest:"
  echo "  results/baselines/verimap/r1_r5/manifest.json"
  exit 1
fi

echo "Found VeriMAP artifacts:"
echo "  results/baselines/verimap/r1_r5/aggregate_by_dataset_model.json"
echo "  results/baselines/verimap/r1_r5/manifest.json"

python -m py_compile scripts/baselines/verimap/overlay_verimap_into_pcgmas.py
python -m py_compile scripts/baselines/verimap/verify_verimap_adapter.py
python -m py_compile scripts/figures/make_paper_figures.py
python -m py_compile scripts/tables/make_paper_tables.py
python -m py_compile scripts/tables/collect_paper_metrics.py
python -m py_compile scripts/tables/repair_paper_metrics_metadata.py
python -m py_compile scripts/tables/validate_paper_metrics.py

# Start from fresh PCG-MAS metrics.
python scripts/tables/collect_paper_metrics.py
python scripts/tables/repair_paper_metrics_metadata.py

# Preserve independent existing overlays that do not rebuild final artifacts.
# Do not call other baseline merge scripts here; VeriMAP is appendix-only.
if [[ -f results/baselines/agentrr/r1_r5/aggregate_by_dataset_model.json && -x scripts/baselines/agentrr/overlay_agentrr_into_pcgmas.py ]]; then
  echo ""
  echo "AgentRR artifacts detected; overlaying AgentRR before VeriMAP."
  python scripts/baselines/agentrr/overlay_agentrr_into_pcgmas.py
fi

echo ""
echo "Overlaying real VeriMAP artifacts into paper_metrics.jsonl."
python scripts/baselines/verimap/overlay_verimap_into_pcgmas.py

echo ""
echo "Restoring prior SOTA overlays before rebuilding appendix hero."

if [[ -f scripts/baselines/shieldagent/overlay_shieldagent_into_pcgmas.py ]] && [[ -f results/baselines/shieldagent/r1_r5/aggregate_by_dataset_model.json ]]; then
  python scripts/baselines/shieldagent/overlay_shieldagent_into_pcgmas.py
fi

if [[ -f scripts/baselines/agentrr/overlay_agentrr_into_pcgmas.py ]] && [[ -f results/baselines/agentrr/r1_r5/aggregate_by_dataset_model.json ]]; then
  python scripts/baselines/agentrr/overlay_agentrr_into_pcgmas.py
fi

python scripts/baselines/verimap/overlay_verimap_into_pcgmas.py


python scripts/tables/validate_paper_metrics.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --allow-partial

python scripts/baselines/verimap/verify_verimap_adapter.py

rm -rf results/figures_staged_selected results/tables/tex_staged_selected
mkdir -p results/figures_staged_selected results/tables/tex_staged_selected results/figures results/tables/tex

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

# VeriMAP is appendix-only. It may update appendix_hero_v4 after overlaying
# VeriMAP fields, but it must not delete or overwrite main R1-R5 figures,
# optional/additional figures, or unrelated tables already present in the
# final artifact directories.
figs = ["appendix_hero_v4"]
tabs = []

copied_figures = []
missing_figures = []
for stem in figs:
    for ext in [".pdf", ".png"]:
        src = Path("results/figures_staged_selected") / f"{stem}{ext}"
        dst = Path("results/figures") / f"{stem}{ext}"
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_figures.append(str(dst))
        else:
            missing_figures.append(stem + ext)

copied_tables = []
missing_tables = []
for stem in tabs:
    src = Path("results/tables/tex_staged_selected") / f"{stem}.tex"
    dst = Path("results/tables/tex") / f"{stem}.tex"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_tables.append(str(dst))
    else:
        missing_tables.append(stem + ".tex")

preserved_patterns = [
    "intro_hero_v4",
    "r1_audit_decomposition_v4",
    "r1_five_channel_audit",
    "r2_redundancy_surface_v4",
    "r3_open_mixed",
    "r3_responsibility_v4",
    "r4_control_frontier_v4",
    "r4_privacy_frontier",
    "r5_overhead_v4",
    "r5_scaling",
    "ablations",
]
preserved_existing = []
for stem in preserved_patterns:
    for ext in [".pdf", ".png"]:
        q = Path("results/figures") / f"{stem}{ext}"
        if q.exists():
            preserved_existing.append(str(q))

manifest = {
    "copied_figures": copied_figures,
    "missing_figures": missing_figures,
    "copied_tables": copied_tables,
    "missing_tables": missing_tables,
    "preserved_existing_figures": preserved_existing,
    "note": "VeriMAP is appendix-only: copied appendix_hero_v4 only; preserved all other final figures/tables.",
}
Path("results/audit").mkdir(parents=True, exist_ok=True)
Path("results/audit/verimap_merge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo ""
echo "Final figures:"
find results/figures -maxdepth 1 -type f | sort

echo ""
echo "Final LaTeX tables:"
find results/tables/tex -maxdepth 1 -type f | sort

echo "VERIMAP_MERGED_INTO_PCGMAS_COMPLETE"
