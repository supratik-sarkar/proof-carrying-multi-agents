#!/usr/bin/env bash

echo "AgentRR-style record/replay baseline runner"
echo "This is an AgentRR-style replay/check adapter over PCG-MAS baseline records."

PCG_ROOT="$(pwd)"

printf "AgentRR virtual environment name [default: agentrr]: "
read -r AGENTRR_VENV_NAME
AGENTRR_VENV_NAME="${AGENTRR_VENV_NAME:-agentrr}"

printf "PCG-MAS virtual environment name to return to later [default: multi-agents]: "
read -r PCG_VENV_NAME
PCG_VENV_NAME="${PCG_VENV_NAME:-multi-agents}"

AGENTRR_VENV="$PCG_ROOT/.venvs/$AGENTRR_VENV_NAME"
PCG_VENV="$PCG_ROOT/$PCG_VENV_NAME"
AGENTRR_WORK_REPO="$PCG_ROOT/.sota_src/agentrr_author"

if [[ ! -d "$AGENTRR_VENV" ]]; then
  echo "AgentRR venv not found: $AGENTRR_VENV"
  echo "Run setup first:"
  echo "  bash scripts/baselines/agentrr/setup_agentrr.sh"
  exit 1
fi

deactivate 2>/dev/null || true
source "$AGENTRR_VENV/bin/activate"

export PCG_ROOT
export AGENTRR_WORK_REPO
export AGENTRR_VENV
export PYTHONPATH="$AGENTRR_WORK_REPO:$AGENTRR_WORK_REPO/agent_rr:$PCG_ROOT:${PYTHONPATH:-}"

echo ""
echo "Using AgentRR Python:"
which python
python --version

echo ""
echo "AgentRR backend mode:"
echo "  1. openai   - OpenAI replay/check-function agent"
echo "  2. hf_local - local Hugging Face replay/check-function agent"
read -r -p "Choose backend [default: 2]: " BACKEND_CHOICE
BACKEND_CHOICE="${BACKEND_CHOICE:-2}"

if [[ "$BACKEND_CHOICE" == "1" || "$BACKEND_CHOICE" == "openai" ]]; then
  BACKEND_MODE="openai"
elif [[ "$BACKEND_CHOICE" == "2" || "$BACKEND_CHOICE" == "hf_local" ]]; then
  BACKEND_MODE="hf_local"
else
  echo "Invalid backend choice: $BACKEND_CHOICE. Use 1=openai or 2=hf_local."
  return 1 2>/dev/null || exit 1
fi

if [[ "$BACKEND_MODE" == "openai" ]]; then
  read -r -s -p "OPENAI_API_KEY [hidden; required]: " OPENAI_API_KEY_INPUT
  echo ""
  export OPENAI_API_KEY="${OPENAI_API_KEY_INPUT:-${OPENAI_API_KEY:-}}"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for backend=openai."
    return 1 2>/dev/null || exit 1
  fi
  read -r -p "OpenAI model [default: gpt-4o-mini]: " OPENAI_MODEL
  OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
  export OPENAI_MODEL
else
  read -r -s -p "HF_TOKEN [hidden; optional/public, required for gated models]: " HF_TOKEN_INPUT
  echo ""
  if [[ -n "$HF_TOKEN_INPUT" ]]; then
    export HF_TOKEN="$HF_TOKEN_INPUT"
  fi
fi

echo "Choose AgentRR benchmark cells using dataset:model syntax."
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

python scripts/baselines/agentrr/run_agentrr_r1_r5.py \
  --pairs "$CELLS" \
  --seeds "$SEEDS" \
  --n-examples "$N_EXAMPLES" \
  --backend-mode "$BACKEND_MODE" \
  --list-cells

printf "Proceed with AgentRR-style replay/check agent run? y/n [default: y]: "
read -r PROCEED
PROCEED="${PROCEED:-y}"

if [[ "$PROCEED" != "y" && "$PROCEED" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

python scripts/baselines/agentrr/run_agentrr_r1_r5.py \
  --pairs "$CELLS" \
  --seeds "$SEEDS" \
  --n-examples "$N_EXAMPLES" \
  --backend-mode "$BACKEND_MODE"

python - <<'PY'
import json
from pathlib import Path

root = Path("results/baselines/agentrr/r1_r5")
manifest = root / "manifest.json"
agg = root / "aggregate_by_dataset_model.json"

if not manifest.exists():
    raise SystemExit("Missing AgentRR manifest.json")
if not agg.exists():
    raise SystemExit("Missing AgentRR aggregate_by_dataset_model.json")

m = json.loads(manifest.read_text())
summary_files = [Path(p) for p in m.get("summary_files", [])]
if not summary_files:
    raise SystemExit("No AgentRR summary files recorded.")

required = [
    "input.jsonl",
    "agentrr_replay_check.jsonl",
    "agentrr_hero_metrics.json",
    "r5_overhead.json",
    "summary.json",
]
missing = []
for s in summary_files:
    # summary_files are authoritative. Do not reconstruct paths from n_records
    # or n_decisions because the hardened runner has multiple corruption
    # decisions per original record.
    s = Path(s)
    if not s.exists():
        missing.append(str(s))
        continue
    for leaf in required:
        q = s.parent / leaf
        if not q.exists():
            missing.append(str(q))
if missing:
    raise SystemExit("Missing AgentRR artifacts:\n" + "\n".join(missing))

print("AGENTRR_REPRODUCIBILITY_RUN_VERIFIED")
print(json.dumps(json.loads(agg.read_text()), indent=2, sort_keys=True))
PY

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

echo "AGENTRR_INTERACTIVE_RUN_COMPLETE"
