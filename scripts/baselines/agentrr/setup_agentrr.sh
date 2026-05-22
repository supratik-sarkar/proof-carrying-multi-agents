#!/usr/bin/env bash

# =====================================================================
# AgentRR setup for PCG-MAS
# =====================================================================
#
# AgentRR best-available source:
#   https://github.com/IPADS-SAI/MobiAgent
#
# The official AgentRR implementation lives under MobiAgent/agent_rr and is
# mobile-task oriented. For PCG-MAS, we keep that source isolated and run a
# PCG-MAS adapter over baseline_inputs.
# =====================================================================

echo "AgentRR setup for PCG-MAS"

PCG_ROOT="$(pwd)"

printf "Path to local MobiAgent clone [default: empty => clone into external path]: "
read -r AGENTRR_SOURCE_REPO

if [[ -z "$AGENTRR_SOURCE_REPO" ]]; then
  printf "External parent directory for MobiAgent clone [default: $HOME/Desktop/My_Git]: "
  read -r EXT_PARENT
  EXT_PARENT="${EXT_PARENT:-$HOME/Desktop/My_Git}"
  mkdir -p "$EXT_PARENT"
  AGENTRR_SOURCE_REPO="$EXT_PARENT/MobiAgent"
  if [[ ! -d "$AGENTRR_SOURCE_REPO" ]]; then
    git clone https://github.com/IPADS-SAI/MobiAgent.git "$AGENTRR_SOURCE_REPO"
  fi
fi

printf "AgentRR virtual environment name [default: agentrr]: "
read -r AGENTRR_VENV_NAME
AGENTRR_VENV_NAME="${AGENTRR_VENV_NAME:-agentrr}"

AGENTRR_WORK_REPO="$PCG_ROOT/.sota_src/agentrr_author"
AGENTRR_VENV="$PCG_ROOT/.venvs/$AGENTRR_VENV_NAME"

if [[ ! -d "$AGENTRR_SOURCE_REPO" ]]; then
  echo "MobiAgent source repo not found: $AGENTRR_SOURCE_REPO"
  exit 1
fi

if [[ ! -d "$AGENTRR_SOURCE_REPO/agent_rr" ]]; then
  echo "Expected agent_rr/ not found inside: $AGENTRR_SOURCE_REPO"
  exit 1
fi

if command -v python3.10 >/dev/null 2>&1; then
  PYBIN="python3.10"
elif command -v python3.11 >/dev/null 2>&1; then
  PYBIN="python3.11"
else
  echo "Need python3.10 or python3.11 for AgentRR venv."
  exit 1
fi

echo "Using $PYBIN:"
$PYBIN --version

deactivate 2>/dev/null || true

mkdir -p "$PCG_ROOT/.venvs"
mkdir -p "$PCG_ROOT/.sota_src"
mkdir -p "$PCG_ROOT/results/baselines/agentrr/runtime"
mkdir -p "$PCG_ROOT/results/baselines/agentrr/r1_r5"
mkdir -p "$PCG_ROOT/results/tables/csv/agentrr_outputs"

printf "Recreate AgentRR venv and scratch copy? y/n [default: y]: "
read -r RECREATE
RECREATE="${RECREATE:-y}"

if [[ "$RECREATE" == "y" || "$RECREATE" == "Y" ]]; then
  rm -rf "$AGENTRR_VENV"
  rm -rf "$AGENTRR_WORK_REPO"
fi

if [[ ! -d "$AGENTRR_WORK_REPO" ]]; then
  rsync -a \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '.venvs' \
    --exclude 'node_modules' \
    --exclude '.DS_Store' \
    --exclude '*.pyc' \
    "$AGENTRR_SOURCE_REPO/" \
    "$AGENTRR_WORK_REPO/"
fi

if [[ ! -d "$AGENTRR_VENV" ]]; then
  "$PYBIN" -m venv "$AGENTRR_VENV"
fi

source "$AGENTRR_VENV/bin/activate"

echo ""
echo "Using AgentRR Python:"
which python
python --version

python -m pip install --upgrade pip setuptools wheel

REQ="$AGENTRR_WORK_REPO/agent_rr/requirements-agentrr.txt"
if [[ -f "$REQ" ]]; then
  echo "Installing official AgentRR requirements from $REQ"
  python -m pip install -r "$REQ" || true
else
  echo "Official AgentRR requirements not found; installing lightweight adapter dependencies."
fi

python -m pip install numpy pandas tqdm pyyaml

python -m pip freeze | sort > "$PCG_ROOT/results/baselines/agentrr/runtime/agentrr.freeze.txt"

cat > "$PCG_ROOT/results/baselines/agentrr/runtime/setup_env.sh" <<ENV
export PCG_ROOT="$PCG_ROOT"
export AGENTRR_WORK_REPO="$AGENTRR_WORK_REPO"
export AGENTRR_VENV="$AGENTRR_VENV"
export PYTHONPATH="$AGENTRR_WORK_REPO:$AGENTRR_WORK_REPO/agent_rr:\${PYTHONPATH:-}"
ENV

cd "$PCG_ROOT"

python -m py_compile scripts/baselines/agentrr/run_agentrr_r1_r5.py

echo ""
echo "AGENTRR_SETUP_COMPLETE"
echo "Source scratch: $AGENTRR_WORK_REPO"
echo "Venv: $AGENTRR_VENV"
