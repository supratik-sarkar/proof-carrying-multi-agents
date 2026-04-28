#!/usr/bin/env bash
# Run the 5 LLMs that fit on M-series MacBook unified memory.
# Each LLM × 5 R-experiments × 8 datasets = 200 cell.json files per LLM.
# Total wall time on M4 Pro 32 GB: ~30-40 hours per LLM at n=200.
# For sanity-check before the big run: pass SMOKE=1 to use n=20.
#
# Usage:
#   scripts/run_local_llms.sh                  # full run, all 5 LLMs
#   SMOKE=1 scripts/run_local_llms.sh          # quick sanity, n=20
#   LLMS="phi-3.5-mini Gemma-2-9b-it" scripts/run_local_llms.sh   # subset
#   DATASETS="hotpotqa fever" scripts/run_local_llms.sh           # subset

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"

# Activate the project venv if it exists
if [[ -f multi-agents/bin/activate ]]; then
  source multi-agents/bin/activate
fi

# Defaults — override via env
LLMS="${LLMS:-phi-3.5-mini qwen2.5-7B deepseek-llm-7b-chat Llama-3.1-8B Gemma-2-9b-it}"
DATASETS="${DATASETS:-hotpotqa twowiki toolbench fever pubmedqa tatqa weblinx synthetic}"
SEEDS="${SEEDS:-0 1 2}"
N_EXAMPLES="${N_EXAMPLES:-200}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  N_EXAMPLES=20
  SEEDS="0"
  echo "SMOKE=1: using n=$N_EXAMPLES, seed=$SEEDS"
fi

# HF model id for each short name
declare -A LLM_REPO=(
  ["phi-3.5-mini"]="microsoft/Phi-3.5-mini-instruct"
  ["qwen2.5-7B"]="Qwen/Qwen2.5-7B-Instruct"
  ["deepseek-llm-7b-chat"]="deepseek-ai/deepseek-llm-7b-chat"
  ["Llama-3.1-8B"]="meta-llama/Llama-3.1-8B-Instruct"
  ["Gemma-2-9b-it"]="google/gemma-2-9b-it"
)

# Mapping from R-experiment to its run script + canonical config
declare -A R_SCRIPT=(
  ["r1"]="run_r1_checkability.py"
  ["r2"]="run_r2_redundancy.py"
  ["r3"]="run_r3_responsibility.py"
  ["r4"]="run_r4_risk_privacy.py"
  ["r5"]="run_r5_overhead.py"
)
declare -A R_CFG_BASE=(
  ["r1"]="r1_hotpotqa.yaml"
  ["r2"]="r2_redundancy.yaml"
  ["r3"]="r3_responsibility.yaml"
  ["r4"]="r4_risk.yaml"
  ["r5"]="r5_overhead.yaml"
)

total_cells=0
for llm in $LLMS; do
  repo="${LLM_REPO[$llm]:-}"
  if [[ -z "$repo" ]]; then
    echo "[skip] unknown LLM short-name: $llm" >&2
    continue
  fi
  for r_id in r1 r2 r3 r4 r5; do
    for ds in $DATASETS; do
      # Prefer per-dataset config when present
      cfg="configs/${r_id}_${ds}.yaml"
      if [[ ! -f "$cfg" ]]; then
        cfg="configs/${R_CFG_BASE[$r_id]}"
      fi
      script="scripts/${R_SCRIPT[$r_id]}"
      echo ""
      echo ">>> $r_id · $llm · $ds  (cfg=$cfg, n=$N_EXAMPLES, seeds=$SEEDS)"
      python "$script" \
        --config "$cfg" \
        --seeds $SEEDS \
        --n-examples "$N_EXAMPLES" \
        --backend hf_local \
        --model-name "$repo" \
        --dataset-override "$ds" \
        --track-cell "${llm}:${ds}"
      total_cells=$((total_cells + 1))
    done
  done
done

echo ""
echo "=========================================="
echo "Done: $total_cells (LLM × dataset × R) cells written."
echo "Run scripts/pick_top_k.py to swap diverse-coverage for top-3 selections."
