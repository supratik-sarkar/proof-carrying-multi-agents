#!/usr/bin/env bash
set -euo pipefail

echo "VeriMAP-Adapter interactive baseline runner"
echo "This is a VeriMAP-style verification-aware planning adapter over PCG-MAS baseline records."

PCG_ROOT="$(pwd)"

printf "VeriMAP virtual environment name [default: verimap]: "
read -r VERIMAP_VENV_NAME
VERIMAP_VENV_NAME="${VERIMAP_VENV_NAME:-verimap}"

printf "PCG-MAS virtual environment name to return to later [default: multi-agents]: "
read -r PCG_VENV_NAME
PCG_VENV_NAME="${PCG_VENV_NAME:-multi-agents}"

VERIMAP_VENV="$PCG_ROOT/.venvs/$VERIMAP_VENV_NAME"
PCG_VENV="$PCG_ROOT/$PCG_VENV_NAME"
VERIMAP_WORK_REPO="$PCG_ROOT/.sota_src/verimap_author"

if [[ ! -d "$VERIMAP_VENV" ]]; then
  echo "VeriMAP venv not found: $VERIMAP_VENV"
  echo "Run setup first:"
  echo "  bash scripts/baselines/verimap/setup_verimap.sh"
  exit 1
fi

deactivate 2>/dev/null || true
source "$VERIMAP_VENV/bin/activate"

export PCG_ROOT
export VERIMAP_WORK_REPO
export VERIMAP_VENV
export PYTHONPATH="$VERIMAP_WORK_REPO:$PCG_ROOT:${PYTHONPATH:-}"

echo ""
echo "Using VeriMAP Python:"
which python
python --version

echo ""
echo "VeriMAP backend mode:"
echo "  1. openai   - OpenAI verification-aware planning/check agent"
echo "  2. hf_local - local Hugging Face verification-aware planning/check agent"
printf "Choose backend [default: 2]: "
read -r BACKEND_CHOICE
BACKEND_CHOICE="${BACKEND_CHOICE:-2}"

case "$BACKEND_CHOICE" in
  1|openai|OPENAI)
    BACKEND_MODE="openai"
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      printf "OPENAI_API_KEY [hidden; required]: "
      read -rs OPENAI_API_KEY
      printf "
"
      export OPENAI_API_KEY
    fi
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "OPENAI_API_KEY is required when backend=openai."
      exit 1
    fi
    printf "OpenAI model [default: gpt-4o-mini]: "
    read -r VERIMAP_OPENAI_MODEL
    VERIMAP_OPENAI_MODEL="${VERIMAP_OPENAI_MODEL:-gpt-4o-mini}"
    export VERIMAP_OPENAI_MODEL
    ;;
  2|hf_local|HF_LOCAL|hf|local)
    BACKEND_MODE="hf_local"
    if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
      printf "HF_TOKEN [hidden; optional/public, required for gated models]: "
      read -rs HF_TOKEN_INPUT
      printf "
"
      if [[ -n "$HF_TOKEN_INPUT" ]]; then
        export HF_TOKEN="$HF_TOKEN_INPUT"
        export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN_INPUT"
      fi
    fi
    # Avoid the Phi-3.5 DynamicCache incompatibility seen with some
    # transformers/remote-code combinations. The runner should also pass
    # use_cache=False, but this env flag gives the adapter a second safe hook.
    export VERIMAP_HF_DISABLE_CACHE="${VERIMAP_HF_DISABLE_CACHE:-1}"
    export VERIMAP_HF_ATTN_IMPL="${VERIMAP_HF_ATTN_IMPL:-eager}"
    ;;
  *)
    echo "Invalid backend choice: $BACKEND_CHOICE. Use 1=openai or 2=hf_local."
    exit 1
    ;;
esac

echo "Choose VeriMAP benchmark cells using dataset:model syntax."
echo "Examples:"
echo "  fever:phi-3.5-mini,hotpotqa:qwen2.5-7b"
echo "  all"
echo ""
echo "If cells='all', dataset:model pairs are inferred from PCG-MAS selected-cell artifacts."
printf "Cells to run [default: all]: "
read -r CELLS
CELLS="${CELLS:-all}"

printf "Seeds, comma-separated [default: 0]: "
read -r SEEDS
SEEDS="${SEEDS:-0}"

printf "Number of examples per dataset/seed [default: 3]: "
read -r N_EXAMPLES
N_EXAMPLES="${N_EXAMPLES:-3}"

python scripts/baselines/verimap/run_verimap_r1_r5.py \
  --pairs "$CELLS" \
  --seeds "$SEEDS" \
  --n-examples "$N_EXAMPLES" \
  --backend-mode "$BACKEND_MODE" \
  --list-cells

printf "Proceed with VeriMAP-Adapter run? y/n [default: y]: "
read -r PROCEED
PROCEED="${PROCEED:-y}"

if [[ "$PROCEED" != "y" && "$PROCEED" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

python scripts/baselines/verimap/run_verimap_r1_r5.py \
  --pairs "$CELLS" \
  --seeds "$SEEDS" \
  --n-examples "$N_EXAMPLES" \
  --backend-mode "$BACKEND_MODE"

python scripts/baselines/verimap/verify_verimap_adapter.py --skip-overlay-check

printf "Switch back to PCG-MAS venv '$PCG_VENV_NAME'? y/n [default: y]: "
read -r SWITCH_BACK
SWITCH_BACK="${SWITCH_BACK:-y}"

if [[ "$SWITCH_BACK" == "y" || "$SWITCH_BACK" == "Y" ]]; then
  deactivate 2>/dev/null || true
  if [[ -d "$PCG_VENV" ]]; then
    source "$PCG_VENV/bin/activate"
    export PYTHONPATH="$PCG_ROOT:${PYTHONPATH:-}"
    which python
    python --version
  fi
fi

echo "VERIMAP_INTERACTIVE_RUN_COMPLETE"
