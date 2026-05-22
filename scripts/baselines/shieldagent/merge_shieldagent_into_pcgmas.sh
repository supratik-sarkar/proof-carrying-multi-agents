#!/usr/bin/env bash
set -euo pipefail

# =====================================================================
# Merge ShieldAgent artifacts into PCG-MAS metrics/figures.
# Independent SOTA-folder version.
#
# Contract:
#   - ShieldAgent remains in scripts/baselines/shieldagent/.
#   - No shared/common baseline folder is required.
#   - Base PCG-MAS metrics may be recollected here because ShieldAgent is
#     the comparative policy baseline, but previously available appendix
#     overlays are reapplied afterward if their artifacts exist.
#   - Final results/figures and results/tables/tex are not deleted.
#   - Full figure/table bundle is rebuilt from final overlaid metrics.
# =====================================================================

echo "Merge ShieldAgent artifacts into PCG-MAS figures/tables"

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

echo ""
echo "Using PCG-MAS Python:"
which python
python --version

SHIELD_ROOT="results/baselines/shieldagent/r1_r5"
SHIELD_AGG="$SHIELD_ROOT/aggregate_by_dataset_model.json"
SHIELD_MANIFEST="$SHIELD_ROOT/manifest.json"

if [[ ! -f "$SHIELD_AGG" ]]; then
  echo "Missing ShieldAgent aggregate file: $SHIELD_AGG"
  echo "Run ShieldAgent first:"
  echo "  bash scripts/baselines/shieldagent/run_shieldagent_interactive.sh"
  exit 1
fi

if [[ ! -f "$SHIELD_MANIFEST" ]]; then
  echo "Missing ShieldAgent manifest file: $SHIELD_MANIFEST"
  exit 1
fi

if [[ ! -f scripts/baselines/shieldagent/overlay_shieldagent_into_pcgmas.py ]]; then
  echo "Missing ShieldAgent overlay script: scripts/baselines/shieldagent/overlay_shieldagent_into_pcgmas.py"
  exit 1
fi

echo ""
echo "Found ShieldAgent artifacts:"
echo "  $SHIELD_AGG"
echo "  $SHIELD_MANIFEST"

echo ""
echo "Compiling required scripts."
python -m py_compile scripts/tables/collect_paper_metrics.py
python -m py_compile scripts/tables/repair_paper_metrics_metadata.py 2>/dev/null || true
python -m py_compile scripts/tables/validate_paper_metrics.py
python -m py_compile scripts/figures/make_paper_figures.py
python -m py_compile scripts/tables/make_paper_tables.py
python -m py_compile scripts/baselines/shieldagent/overlay_shieldagent_into_pcgmas.py
python -m py_compile scripts/baselines/agentrr/overlay_agentrr_into_pcgmas.py 2>/dev/null || true
python -m py_compile scripts/baselines/verimap/overlay_verimap_into_pcgmas.py 2>/dev/null || true

echo ""
echo "Collecting PCG-MAS metrics once for the selected cells."
python scripts/tables/collect_paper_metrics.py

if [[ -f scripts/tables/repair_paper_metrics_metadata.py ]]; then
  python scripts/tables/repair_paper_metrics_metadata.py
fi

echo ""
echo "Overlaying ShieldAgent artifacts."
python scripts/baselines/shieldagent/overlay_shieldagent_into_pcgmas.py

# Reapply independent appendix/intro overlays if their artifacts exist. This
# keeps each SOTA folder independent while preventing a base metric recollect
# from erasing already-run adapter fields.
if [[ -f results/baselines/agentrr/r1_r5/aggregate_by_dataset_model.json && -f scripts/baselines/agentrr/overlay_agentrr_into_pcgmas.py ]]; then
  echo ""
  echo "AgentRR artifacts detected; reapplying AgentRR overlay additively."
  python scripts/baselines/agentrr/overlay_agentrr_into_pcgmas.py
fi

if [[ -f results/baselines/verimap/r1_r5/aggregate_by_dataset_model.json && -f scripts/baselines/verimap/overlay_verimap_into_pcgmas.py ]]; then
  echo ""
  echo "VeriMAP artifacts detected; reapplying VeriMAP overlay additively."
  python scripts/baselines/verimap/overlay_verimap_into_pcgmas.py
fi

echo ""
echo "Validating merged paper metrics."
python scripts/tables/validate_paper_metrics.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --allow-partial

echo ""
echo "Rebuilding final figures and LaTeX tables from fully overlaid paper_metrics.jsonl."
mkdir -p results/figures results/tables/tex results/audit

python scripts/figures/make_paper_figures.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --outdir results/figures \
  --allow-partial

python scripts/tables/make_paper_tables.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --outdir results/tables/tex \
  --allow-partial

python - <<'PY_AUDIT'
import json
from pathlib import Path

rows_path = Path("results/tables/csv/paper_metrics.jsonl")
rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
methods = ["shieldagent", "agentrr", "verimap"]
audit = []
for r in rows:
    audit.append({
        "dataset": r.get("dataset"),
        "model": r.get("model"),
        **{m: bool(r.get(f"{m}_overlay_applied")) for m in methods},
    })
manifest = {
    "rows": audit,
    "note": "Independent ShieldAgent merge rebuilt from final additive overlays.",
}
Path("results/audit").mkdir(parents=True, exist_ok=True)
Path("results/audit/shieldagent_merge_build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, indent=2, sort_keys=True))
PY_AUDIT

echo ""
echo "Final figures:"
find results/figures -maxdepth 1 -type f | sort

echo ""
echo "Final LaTeX tables:"
find results/tables/tex -maxdepth 1 -type f | sort

echo ""
echo "SHIELDAGENT_MERGED_INTO_PCGMAS_COMPLETE"
