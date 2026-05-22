#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || {
  echo "Could not move to PCG-MAS repo root."
  return 1 2>/dev/null || exit 1
}

PCG_ROOT="$PWD"

echo "PCG-MAS interactive runner"
echo "Repo root: $PCG_ROOT"
echo ""

if [ -d "$PCG_ROOT/multi-agents" ]; then
  source "$PCG_ROOT/multi-agents/bin/activate"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "Using active virtual environment: $VIRTUAL_ENV"
else
  echo "Could not find $PCG_ROOT/multi-agents and no venv is active."
  echo "Activate the PCG-MAS venv first, then rerun."
  return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="$PCG_ROOT:${PYTHONPATH:-}"

echo "Using Python:"
which python
python --version
echo ""

printf "HF_TOKEN [hidden; optional; press ENTER to skip/use mock fallback when prompted]: "
read -rs USER_HF_TOKEN
printf "\n"

if [ -n "$USER_HF_TOKEN" ]; then
  export HF_TOKEN="$USER_HF_TOKEN"
  export HUGGINGFACE_HUB_TOKEN="$USER_HF_TOKEN"
  echo "HF token loaded into this shell session."
else
  echo "No HF token exported by wrapper."
fi

echo ""
echo "Enter cells using dataset:model syntax."
echo "Examples:"
echo "  fever:phi-3.5-mini,hotpotqa:qwen2.5-7b,pubmedqa:llama-3.1-8b,tatqa:gemma-2-9b-it"
echo "  toolbench:llama-3.3-70b,weblinx:deepseek-v3"
echo "  all"
echo ""
printf "Cells to run [default: fever:phi-3.5-mini,hotpotqa:qwen2.5-7b,pubmedqa:llama-3.1-8b,tatqa:gemma-2-9b-it]: "
read -r PCG_CELLS
PCG_CELLS="${PCG_CELLS:-fever:phi-3.5-mini,hotpotqa:qwen2.5-7b,pubmedqa:llama-3.1-8b,tatqa:gemma-2-9b-it}"

echo ""
echo "Experiment choices:"
echo "  r1"
echo "  r2"
echo "  r3"
echo "  r4"
echo "  r5"
echo "  r1-r5"
echo "  all"
echo ""
echo "Definitions:"
echo "  r1-r5 = main R1, R2, R3, R4, R5"
echo "  all   = R1-R5 main. Optional/additional figures are built only when their required JSONL inputs exist."
echo ""
printf "Experiments to run [default: r1]: "
read -r PCG_EXPERIMENT_MODE
PCG_EXPERIMENT_MODE="${PCG_EXPERIMENT_MODE:-r1}"

printf "Seeds, comma-separated [default: 0]: "
read -r PCG_SEEDS
PCG_SEEDS="${PCG_SEEDS:-0}"

printf "Number of examples per cell [default: 3]: "
read -r PCG_N_EXAMPLES
PCG_N_EXAMPLES="${PCG_N_EXAMPLES:-3}"

printf "Backend [default: hf_local]: "
read -r PCG_BACKEND
PCG_BACKEND="${PCG_BACKEND:-hf_local}"

printf "Clean existing PCG-MAS figures/tables/paper_metrics before run? y/n [default: y]: "
read -r PCG_CLEAN
PCG_CLEAN="${PCG_CLEAN:-y}"

python - <<PY
import json
from pathlib import Path

raw_cells = """$PCG_CELLS""".strip()
mode = """$PCG_EXPERIMENT_MODE""".strip().lower()
seeds = """$PCG_SEEDS""".strip()
n_examples = int("""$PCG_N_EXAMPLES""".strip())
backend = """$PCG_BACKEND""".strip()

MODEL_ALIASES = {
    "phi-3.5-mini": "phi-3.5-mini",
    "phi35": "phi-3.5-mini",

    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen/qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",

    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",

    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",

    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "google/gemma-2-9b-it": "google/gemma-2-9b-it",

    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/deepseek-v3": "deepseek-ai/DeepSeek-V3",
}

DATASET_ALIASES = {
    "tat-qa": "tatqa",
    "tatqa": "tatqa",
    "hotpot": "hotpotqa",
    "hotpotqa": "hotpotqa",
    "fever": "fever",
    "pubmedqa": "pubmedqa",
    "toolbench": "toolbench",
    "weblinx": "weblinx",
    "2wiki": "2wikimultihopqa",
    "twowiki": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
}

EXPERIMENT_MODES = {
    "r1": ["r1"],
    "r2": ["r2"],
    "r3": ["r3"],
    "r4": ["r4"],
    "r5": ["r5"],
    "r1-r5": ["r1", "r2", "r3", "r4", "r5"],
    "r1_r5": ["r1", "r2", "r3", "r4", "r5"],
    "main": ["r1", "r2", "r3", "r4", "r5"],
    "all": [
        "r1", "r2", "r3", "r4", "r5",
        "ablations",
        "r1_five_channel_audit",
        "r3_open_mixed",
        "r4_privacy_frontier",
        "r5_scaling",
    ],
}

ALL_CELLS = [
    ("fever", "phi-3.5-mini"),
    ("hotpotqa", "Qwen/Qwen2.5-7B-Instruct"),
    ("pubmedqa", "meta-llama/Llama-3.1-8B-Instruct"),
    ("tatqa", "google/gemma-2-9b-it"),
    ("toolbench", "meta-llama/Llama-3.3-70B-Instruct"),
    ("weblinx", "deepseek-ai/DeepSeek-V3"),
]

def norm_dataset(x):
    y = x.strip().lower().replace("_", "-")
    return DATASET_ALIASES.get(y, y)

def norm_model(x):
    y = x.strip().lower().replace("_", "-")
    return MODEL_ALIASES.get(y, x.strip())

if raw_cells.lower() == "all":
    cells = ALL_CELLS
else:
    cells = []
    for part in raw_cells.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"Invalid cell '{part}'. Expected dataset:model.")
        d, m = part.split(":", 1)
        cells.append((norm_dataset(d), norm_model(m)))

if not cells:
    raise SystemExit("No cells selected.")

if mode not in EXPERIMENT_MODES:
    raise SystemExit(f"Invalid experiment mode '{mode}'. Valid: {sorted(EXPERIMENT_MODES)}")

experiments = EXPERIMENT_MODES[mode]
hero_allowed = mode in {"r1-r5", "r1_r5", "main", "all"}

hero_primary = [
    ("fever", "phi-3.5-mini"),
    ("tatqa", "google/gemma-2-9b-it"),
    ("toolbench", "meta-llama/Llama-3.3-70B-Instruct"),
    ("weblinx", "deepseek-ai/DeepSeek-V3"),
]
hero_fallback = [
    ("hotpotqa", "Qwen/Qwen2.5-7B-Instruct"),
    ("pubmedqa", "meta-llama/Llama-3.1-8B-Instruct"),
]
hero_eligible_order = hero_primary + hero_fallback

selected_set = set(cells)
hero_cells = []
for c in hero_eligible_order:
    if c in selected_set and c not in hero_cells:
        hero_cells.append(c)
    if len(hero_cells) >= 4:
        break

main_r_cells = cells[:3]
additional_cells = cells[:2]

def hero_message(name):
    if hero_cells and hero_allowed:
        return f"{name} will use {len(hero_cells)} qualifying cell(s)."
    if hero_cells and not hero_allowed:
        return f"{name} will be skipped because it is only built for r1-r5/main/all experiment mode."
    return f"{name} will be skipped because no qualifying cells were selected."

payload = {
    "cells": [{"dataset": d, "model": m} for d, m in cells],
    "cells_cli": ",".join(f"{d}:{m}" for d, m in cells),
    "experiments": experiments,
    "experiments_cli": ",".join(experiments),
    "experiment_mode": mode,
    "seeds": seeds,
    "n_examples": n_examples,
    "backend": backend,
    "figure_policy": {
        "intro_hero_v4": {
            "selected_cells": [{"dataset": d, "model": m} for d, m in hero_cells],
            "max_cells": 4,
            "build": bool(hero_cells) and hero_allowed,
            "message": hero_message("intro_hero_v4"),
        },
        "appendix_hero_v4": {
            "selected_cells": [{"dataset": d, "model": m} for d, m in hero_cells],
            "max_cells": 4,
            "build": bool(hero_cells) and hero_allowed,
            "message": hero_message("appendix_hero_v4"),
        },
        "main_r1_r5": {
            "selected_cells": [{"dataset": d, "model": m} for d, m in main_r_cells],
            "max_cells": 3,
            "message": (
                f"main R1-R5 figures will use first {len(main_r_cells)} selected cell(s)."
                + (f" {len(cells)-3} selected cell(s) will be excluded from main R1-R5 figures." if len(cells) > 3 else "")
            ),
        },
        "additional_figures": {
            "selected_cells": [{"dataset": d, "model": m} for d, m in additional_cells],
            "max_cells": 2,
            "sota_allowed": False,
            "message": (
                f"additional figures will use first {len(additional_cells)} selected cell(s), PCG-MAS vs NoCert only."
                + (f" {len(cells)-2} selected cell(s) will be excluded from additional figures." if len(cells) > 2 else "")
            ),
        },
    },
}

Path("results/audit").mkdir(parents=True, exist_ok=True)
Path("results/audit/pcgmas_selected_cells.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

print("PCG-MAS selected cell policy:")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [ $? -ne 0 ]; then
  echo "Cell/experiment parsing failed."
  return 1 2>/dev/null || exit 1
fi

echo ""
echo "Resolved PCG-MAS run plan:"
python - <<'PY'
import json
from pathlib import Path

p = json.loads(Path("results/audit/pcgmas_selected_cells.json").read_text())

print("Cells:")
for i, c in enumerate(p["cells"], start=1):
    print(f"  {i}/{len(p['cells'])} {c['dataset']}:{c['model']}")
print("Experiments:", ",".join(p["experiments"]))
print("Seeds:", p["seeds"])
print("n_examples:", p["n_examples"])
print("Backend:", p["backend"])
print("")
for name, spec in p["figure_policy"].items():
    print(f"{name}: {spec['message']}")
PY

echo ""
echo "Compiling PCG-MAS scripts."

python -m py_compile scripts/runs/run_matrix.py
python -m py_compile scripts/tables/collect_paper_metrics.py
python -m py_compile scripts/tables/validate_paper_metrics.py
python -m py_compile scripts/tables/repair_paper_metrics_metadata.py
python -m py_compile scripts/figures/build_all_figures.py
python -m py_compile scripts/figures/make_paper_figures.py
python -m py_compile scripts/tables/build_all_tables.py
python -m py_compile scripts/tables/make_paper_tables.py

if [ $? -ne 0 ]; then
  echo "Compilation failed."
  return 1 2>/dev/null || exit 1
fi

if [ "$PCG_CLEAN" = "y" ] || [ "$PCG_CLEAN" = "Y" ]; then
  echo "Cleaning PCG-MAS generated figures/tables/paper metrics."
  rm -rf results/figures
  rm -rf results/figures_staged_selected
  rm -rf results/tables/tex
  rm -rf results/tables/tex_staged_selected
  rm -rf results/tables/csv/experiment_json
  rm -f results/tables/csv/paper_metrics.jsonl
  rm -f results/tables/csv/paper_metrics_with_shieldagent.jsonl
  rm -f results/tables/csv/paper_metrics_with_shieldagent_wide.jsonl
  rm -f results/tables/csv/shieldagent_metrics_for_figures.jsonl
  rm -f results/tables/csv/shieldagent_merge_report.json
  rm -f results/tables/csv/shieldagent_wide_merge_report.json

  mkdir -p results/figures
  mkdir -p results/tables/tex
  mkdir -p results/tables/csv
  mkdir -p results/tables/csv/baseline_inputs
  mkdir -p results/tables/csv/experiment_json
fi

echo "Removing stale baseline_inputs for selected cells only."

python - <<'PY'
import json
from pathlib import Path

policy = json.loads(Path("results/audit/pcgmas_selected_cells.json").read_text())
datasets = [x["dataset"].lower() for x in policy["cells"]]

root = Path("results/tables/csv/baseline_inputs")
root.mkdir(parents=True, exist_ok=True)

for p in sorted(root.glob("*baseline_inputs.jsonl")):
    name = p.name.lower()
    if any(d in name for d in datasets):
        print("DELETE", p)
        p.unlink()
PY

echo ""
echo "Running PCG-MAS cells one by one."

python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

policy = json.loads(Path("results/audit/pcgmas_selected_cells.json").read_text())

experiments = [str(x) for x in policy["experiments"]]
run_experiments = [x for x in experiments if x in {"r1", "r2", "r3", "r4", "r5"}]
if not run_experiments:
    run_experiments = ["r1"]
experiments_display = ",".join(run_experiments)
experiments_requested_display = ",".join(experiments)
seeds = str(policy["seeds"])
n_examples = str(policy["n_examples"])
backend = str(policy["backend"])

env = os.environ.copy()

for i, cell in enumerate(policy["cells"], start=1):
    dataset = cell["dataset"]
    model = cell["model"]

    print("")
    print(f"PCG-MAS cell {i}/{len(policy['cells'])}: {dataset}:{model}")
    print(f"  experiments={experiments_display}")
    if experiments_requested_display != experiments_display:
        print(f"  requested_experiments={experiments_requested_display}")
    print(f"  seeds={seeds}")
    print(f"  n_examples={n_examples}")
    print(f"  backend={backend}")
    print("")

    cmd = [
        sys.executable,
        "scripts/runs/run_matrix.py",
        "--allow-full-run",
        "--allow-dataset-fallback",
        "--n-examples", n_examples,
        "--seeds", seeds,
        "--datasets", dataset,
        "--models", model,
        "--experiments", *run_experiments,
        "--backend", backend,
    ]

    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise SystemExit(f"PCG-MAS cell failed: {dataset}:{model}")

print("")
print("PCGMAS_SELECTED_CELLS_RUN_COMPLETE")
PY

if [ $? -ne 0 ]; then
  echo "PCG-MAS selected-cell run failed."
  return 1 2>/dev/null || exit 1
fi

echo ""
echo "Collecting PCG-MAS metrics."

python scripts/tables/collect_paper_metrics.py

if [ $? -ne 0 ]; then
  echo "collect_paper_metrics.py failed."
  return 1 2>/dev/null || exit 1
fi

echo "Repairing metadata."

python scripts/tables/repair_paper_metrics_metadata.py

if [ $? -ne 0 ]; then
  echo "repair_paper_metrics_metadata.py failed."
  return 1 2>/dev/null || exit 1
fi

echo "Validating partial metrics."

python scripts/tables/validate_paper_metrics.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --allow-partial

if [ $? -ne 0 ]; then
  echo "Metric validation failed."
  return 1 2>/dev/null || exit 1
fi

echo ""
echo "Building PCG-MAS figures and tables with selected-experiment artifact policy."

rm -rf results/figures_staged_selected
rm -rf results/tables/tex_staged_selected
rm -rf results/figures
rm -rf results/tables/tex

mkdir -p results/figures_staged_selected
mkdir -p results/tables/tex_staged_selected
mkdir -p results/figures
mkdir -p results/tables/tex
mkdir -p results/audit

python scripts/figures/make_paper_figures.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --outdir results/figures_staged_selected \
  --allow-partial

if [ $? -ne 0 ]; then
  echo "Figure staging build failed."
  return 1 2>/dev/null || exit 1
fi

python scripts/tables/make_paper_tables.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --outdir results/tables/tex_staged_selected \
  --allow-partial

if [ $? -ne 0 ]; then
  echo "Table staging build failed."
  return 1 2>/dev/null || exit 1
fi

python - <<'INNERPY'
import json
import shutil
from pathlib import Path

policy = json.loads(Path("results/audit/pcgmas_selected_cells.json").read_text())
mode = str(policy.get("experiment_mode", "")).lower()
experiments = set(policy.get("experiments", []))

staged_fig = Path("results/figures_staged_selected")
final_fig = Path("results/figures")
staged_tab = Path("results/tables/tex_staged_selected")
final_tab = Path("results/tables/tex")

final_fig.mkdir(parents=True, exist_ok=True)
final_tab.mkdir(parents=True, exist_ok=True)

main_figures = {
    "r1": {"r1_audit_decomposition_v4"},
    "r2": {"r2_redundancy_surface_v4"},
    "r3": {"r3_responsibility_v4"},
    "r4": {"r4_control_frontier_v4"},
    "r5": {"r5_overhead_v4"},
}

additional_figures = {
    "ablations",
    "r1_five_channel_audit",
    "r3_open_mixed",
    "r4_privacy_frontier",
    "r5_scaling",
}

allowed_figures = set()

if mode in {"r1-r5", "r1_r5", "main", "all"}:
    allowed_figures.update({
        "intro_hero_v4",
        "appendix_hero_v4",
    })

for exp in experiments:
    allowed_figures.update(main_figures.get(exp, set()))

if mode == "all":
    allowed_figures.update(additional_figures)

print("[artifact policy] Final figures allowed:")
for x in sorted(allowed_figures):
    print(" ", x)

copied_figures = []
missing_figures = []
skipped_figures = []

for stem in sorted(allowed_figures):
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

for src in sorted(staged_fig.glob("*")):
    if src.is_file() and src.stem not in allowed_figures:
        skipped_figures.append(str(src))

print("[artifact policy] Copied figures:")
for x in copied_figures:
    print(" ", x)

if missing_figures:
    print("[artifact policy] Missing allowed figures:")
    for x in missing_figures:
        print(" ", x)

print("[artifact policy] Skipped unsupported figures:")
for x in skipped_figures:
    print(" ", x)

table_by_scope = set()

if "r1" in experiments:
    table_by_scope.update({
        "table_replay_drift_covgap",
        "table_r1_r4_combined",
    })

if "r4" in experiments:
    table_by_scope.update({
        "table_r1_r4_combined",
        "table_r4_privacy",
    })

if "r5" in experiments:
    table_by_scope.update({
        "table_cost_overhead_main",
    })

if mode in {"r1-r5", "r1_r5", "main"}:
    table_by_scope.update({
        "table_main_six_summary",
        "table_cost_overhead_main",
        "table_r1_r4_combined",
        "table_replay_drift_covgap",
        "table_appendix_remaining_50_summary",
        "table_appendix_remaining_50_r1r4",
        "table_appendix_remaining_50_cost",
    })

if mode == "all":
    table_by_scope.update({
        "table_main_six_summary",
        "table_cost_overhead_main",
        "table_r1_r4_combined",
        "table_replay_drift_covgap",
        "table_appendix_remaining_50_summary",
        "table_appendix_remaining_50_r1r4",
        "table_appendix_remaining_50_cost",
        "table_ablations",
        "table_r3_open_mixed",
        "table_r4_privacy",
        "table_r5_scaling",
    })

print("[artifact policy] Final tables allowed:")
for x in sorted(table_by_scope):
    print(" ", x)

copied_tables = []
missing_tables = []
skipped_tables = []

for stem in sorted(table_by_scope):
    src = staged_tab / f"{stem}.tex"
    if src.exists():
        dst = final_tab / src.name
        shutil.copy2(src, dst)
        copied_tables.append(str(dst))
    else:
        missing_tables.append(stem)

for src in sorted(staged_tab.glob("*.tex")):
    if src.stem not in table_by_scope:
        skipped_tables.append(str(src))

print("[artifact policy] Copied tables:")
for x in copied_tables:
    print(" ", x)

if missing_tables:
    print("[artifact policy] Missing allowed tables:")
    for x in missing_tables:
        print(" ", x)

print("[artifact policy] Skipped unsupported tables:")
for x in skipped_tables:
    print(" ", x)

manifest = {
    "experiment_mode": mode,
    "experiments": sorted(experiments),
    "allowed_figures": sorted(allowed_figures),
    "copied_figures": copied_figures,
    "missing_figures": missing_figures,
    "skipped_figures": skipped_figures,
    "allowed_tables": sorted(table_by_scope),
    "copied_tables": copied_tables,
    "missing_tables": missing_tables,
    "skipped_tables": skipped_tables,
    "note": "Unsupported figures/tables are not copied to final artifact folders to avoid reference outputs.",
}

Path("results/audit/selected_artifact_policy_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True)
)
INNERPY

echo ""
echo "Generated selected-cell baseline_inputs:"
python - <<'PY'
import json
from pathlib import Path

policy = json.loads(Path("results/audit/pcgmas_selected_cells.json").read_text())
datasets = [x["dataset"].lower() for x in policy["cells"]]

root = Path("results/tables/csv/baseline_inputs")
for p in sorted(root.glob("*baseline_inputs.jsonl")):
    name = p.name.lower()
    if any(d in name for d in datasets):
        print(p)
PY

echo ""
echo "Generated paper metrics:"
cat results/tables/csv/paper_metrics.jsonl

echo ""
echo "Final figures:"
find results/figures -maxdepth 1 -type f | sort

echo ""
echo "Final LaTeX tables:"
find results/tables/tex -maxdepth 1 -type f | sort

echo ""
echo "Artifact policy manifest:"
cat results/audit/selected_artifact_policy_manifest.json

echo ""
echo "PCGMAS_INTERACTIVE_RUN_COMPLETE"
