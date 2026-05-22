#!/usr/bin/env bash
set -euo pipefail

echo "VeriMAP-Adapter setup for PCG-MAS"
echo ""

PCG_ROOT="$(pwd)"

printf "Path to local VeriMAP clone [default: empty => clone into external path]: "
read -r VERIMAP_SOURCE_REPO

if [[ -z "${VERIMAP_SOURCE_REPO}" ]]; then
  printf "External parent directory for VeriMAP clone [default: $HOME/Desktop/My_Git]: "
  read -r EXTERNAL_PARENT
  EXTERNAL_PARENT="${EXTERNAL_PARENT:-$HOME/Desktop/My_Git}"

  mkdir -p "$EXTERNAL_PARENT"
  VERIMAP_SOURCE_REPO="$EXTERNAL_PARENT/veriMAP"

  if [[ ! -d "$VERIMAP_SOURCE_REPO" ]]; then
    echo "Cloning VeriMAP source reference into:"
    echo "  $VERIMAP_SOURCE_REPO"
    git clone https://github.com/megagonlabs/veriMAP.git "$VERIMAP_SOURCE_REPO"
  else
    echo "Using existing VeriMAP source reference:"
    echo "  $VERIMAP_SOURCE_REPO"
  fi
fi

if [[ ! -d "$VERIMAP_SOURCE_REPO" ]]; then
  echo "VeriMAP source path does not exist:"
  echo "  $VERIMAP_SOURCE_REPO"
  exit 1
fi

printf "VeriMAP virtual environment name [default: verimap]: "
read -r VERIMAP_VENV_NAME
VERIMAP_VENV_NAME="${VERIMAP_VENV_NAME:-verimap}"

VERIMAP_WORK_REPO="$PCG_ROOT/.sota_src/verimap_author"
VERIMAP_VENV="$PCG_ROOT/.venvs/$VERIMAP_VENV_NAME"

echo ""
echo "PCG_ROOT=$PCG_ROOT"
echo "VERIMAP_SOURCE_REPO=$VERIMAP_SOURCE_REPO"
echo "VERIMAP_WORK_REPO=$VERIMAP_WORK_REPO"
echo "VERIMAP_VENV=$VERIMAP_VENV"
echo ""

mkdir -p "$PCG_ROOT/.sota_src"
mkdir -p "$PCG_ROOT/.venvs"
mkdir -p "$PCG_ROOT/results/baselines/verimap/runtime"

rm -rf "$VERIMAP_WORK_REPO"
mkdir -p "$VERIMAP_WORK_REPO"

rsync -a \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".venv" \
  --exclude "venv" \
  --exclude ".venvs" \
  --exclude "node_modules" \
  --exclude ".DS_Store" \
  --exclude "*.pyc" \
  "$VERIMAP_SOURCE_REPO/" \
  "$VERIMAP_WORK_REPO/"

deactivate 2>/dev/null || true

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
else
  PYTHON_BIN="python3"
fi

rm -rf "$VERIMAP_VENV"
"$PYTHON_BIN" -m venv "$VERIMAP_VENV"
source "$VERIMAP_VENV/bin/activate"

python --version
which python

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PCG_ROOT/scripts/baselines/verimap/requirements.verimap.runtime.txt"

if [[ -f "$VERIMAP_WORK_REPO/requirements.txt" ]]; then
  echo "Installing VeriMAP author requirements if compatible."
  python -m pip install -r "$VERIMAP_WORK_REPO/requirements.txt" || true
fi

python -m pip freeze | sort > "$PCG_ROOT/results/baselines/verimap/runtime/verimap.freeze.txt"

echo ""
echo "VERIMAP_SETUP_COMPLETE"
echo "Source scratch: $VERIMAP_WORK_REPO"
echo "Venv: $VERIMAP_VENV"
