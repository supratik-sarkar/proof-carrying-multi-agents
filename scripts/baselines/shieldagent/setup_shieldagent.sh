#!/usr/bin/env bash

# =====================================================================
# ShieldAgent / AutoPolicy setup for PCG-MAS
# =====================================================================
#
# Run from the PCG-MAS repo root:
#
#   bash scripts/baselines/shieldagent/setup_shieldagent.sh
#
# This setup:
#   1. asks for the local AutoPolicy clone path
#   2. asks for the ShieldAgent venv name
#   3. copies AutoPolicy into .sota_src/shieldagent_author/
#   4. creates an isolated Python 3.11 venv for ShieldAgent
#   5. installs ShieldAgent runtime dependencies
#   6. validates key imports and CLI entry points
#
# It does not modify the original AutoPolicy clone.
# =====================================================================

echo "ShieldAgent / AutoPolicy setup for PCG-MAS"

PCG_ROOT="$(pwd)"

printf "Path to local AutoPolicy clone [default: $HOME/Desktop/My_Git/ShieldAgent_AutoPolicy]: "
read -r SHIELD_SOURCE_REPO
SHIELD_SOURCE_REPO="${SHIELD_SOURCE_REPO:-$HOME/Desktop/My_Git/ShieldAgent_AutoPolicy}"

printf "ShieldAgent virtual environment name [default: shield-agent]: "
read -r SHIELD_VENV_NAME
SHIELD_VENV_NAME="${SHIELD_VENV_NAME:-shield-agent}"

SHIELD_WORK_REPO="$PCG_ROOT/.sota_src/shieldagent_author"
SHIELD_VENV="$PCG_ROOT/.venvs/$SHIELD_VENV_NAME"

echo ""
echo "PCG_ROOT=$PCG_ROOT"
echo "SHIELD_SOURCE_REPO=$SHIELD_SOURCE_REPO"
echo "SHIELD_WORK_REPO=$SHIELD_WORK_REPO"
echo "SHIELD_VENV=$SHIELD_VENV"

if [[ ! -d "$PCG_ROOT" ]]; then
  echo "PCG-MAS repo root not found: $PCG_ROOT"
  exit 1
fi

if [[ ! -d "$SHIELD_SOURCE_REPO" ]]; then
  echo "AutoPolicy clone not found: $SHIELD_SOURCE_REPO"
  echo ""
  echo "Clone it first, for example:"
  echo "  mkdir -p ~/Desktop/My_Git"
  echo "  git clone https://github.com/BillChan226/AutoPolicy.git ~/Desktop/My_Git/ShieldAgent_AutoPolicy"
  exit 1
fi

if [[ ! -f "$SHIELD_SOURCE_REPO/requirements.txt" ]]; then
  echo "AutoPolicy requirements.txt not found under: $SHIELD_SOURCE_REPO"
  exit 1
fi

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 not found."

  if [[ "$(uname -s)" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      echo "Installing python@3.11 via Homebrew."
      brew install python@3.11

      if [[ -d "/opt/homebrew/opt/python@3.11/bin" ]]; then
        export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"
      elif [[ -d "/usr/local/opt/python@3.11/bin" ]]; then
        export PATH="/usr/local/opt/python@3.11/bin:$PATH"
      fi
    else
      echo "Homebrew is not installed. Install Python 3.11 manually or install Homebrew, then rerun."
      exit 1
    fi
  else
    echo "Install Python 3.11 for the target OS, then rerun."
    exit 1
  fi
fi

python3.11 --version
which python3.11

echo ""
echo "This setup can recreate the ShieldAgent venv and scratch copy."
printf "Recreate ShieldAgent venv and scratch copy? y/n [default: y]: "
read -r RECREATE
RECREATE="${RECREATE:-y}"

deactivate 2>/dev/null || true

mkdir -p "$PCG_ROOT/.venvs"
mkdir -p "$PCG_ROOT/.sota_src"
mkdir -p "$PCG_ROOT/scripts/baselines/shieldagent"
mkdir -p "$PCG_ROOT/results/baselines/shieldagent/runtime"
mkdir -p "$PCG_ROOT/results/baselines/shieldagent/raw"
mkdir -p "$PCG_ROOT/results/baselines/shieldagent/patches"
mkdir -p "$PCG_ROOT/results/baselines/shieldagent/selected_policy_bank"
mkdir -p "$PCG_ROOT/results/tables/csv/shieldagent_inputs"
mkdir -p "$PCG_ROOT/results/tables/csv/shieldagent_outputs"

if [[ "$RECREATE" == "y" || "$RECREATE" == "Y" ]]; then
  echo "Removing old ShieldAgent venv/scratch copy."
  rm -rf "$SHIELD_VENV"
  rm -rf "$SHIELD_WORK_REPO"
fi

if [[ ! -d "$SHIELD_WORK_REPO" ]]; then
  echo "Copying AutoPolicy into local scratch folder."
  rsync -a \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude 'multi-agents' \
    --exclude '.venvs' \
    --exclude 'node_modules' \
    --exclude '.DS_Store' \
    --exclude '*.pyc' \
    "$SHIELD_SOURCE_REPO/" \
    "$SHIELD_WORK_REPO/"
fi

cd "$SHIELD_WORK_REPO" || exit 1
rm -f shield
ln -s agent shield

cd "$PCG_ROOT" || exit 1

if [[ ! -d "$SHIELD_VENV" ]]; then
  echo "Creating ShieldAgent venv."
  python3.11 -m venv "$SHIELD_VENV"
fi

source "$SHIELD_VENV/bin/activate"

echo ""
echo "Using ShieldAgent Python:"
which python
python --version

python -m pip install --upgrade pip setuptools wheel

cat > "$PCG_ROOT/scripts/baselines/shieldagent/requirements.shield-agent.runtime.txt" <<'REQ'
anthropic==0.49.0
beautifulsoup4==4.12.3
bs4
fastmcp==0.4.1
mcp==1.6.0
openai==1.69.0
openai-agents==0.0.7
PyMuPDF==1.25.5
pymupdf4llm==0.0.21
PyPDF2==3.0.1
python-dotenv==1.0.1
pydantic==2.10.4
pydantic-settings==2.8.1
requests==2.32.3
aiohttp==3.11.11
httpx==0.28.1
numpy==1.26.4
pandas==2.2.3
tqdm==4.67.1
PyYAML==6.0.2
lxml==5.3.0
markdownify
html2text
tabulate==0.9.0
tiktoken==0.8.0
jsonpatch==1.33
jsonpointer==3.0.0
tenacity==8.5.0
typing_extensions==4.12.2
rich==13.9.4
typer==0.15.2
uvicorn==0.34.0
fastapi==0.115.12
starlette==0.46.1
sse-starlette==2.2.1
python-multipart==0.0.20
matplotlib==3.10.0
networkx==3.3
termcolor==2.4.0
REQ

python -m pip install -r "$PCG_ROOT/scripts/baselines/shieldagent/requirements.shield-agent.runtime.txt"

echo ""
echo "Running pip check."
python -m pip check || true

python -m pip freeze | sort > "$PCG_ROOT/results/baselines/shieldagent/runtime/shield-agent.freeze.txt"

cd "$SHIELD_WORK_REPO" || exit 1

chmod +x agent/lib/LADR-2009-02A/bin/mace4 2>/dev/null || true
chmod +x agent/lib/LADR-2009-02A/utilities/prover9-mace4 2>/dev/null || true

export PYTHONPATH="$SHIELD_WORK_REPO:$SHIELD_WORK_REPO/agent:${PYTHONPATH:-}"
export PATH="$SHIELD_WORK_REPO/agent/lib/LADR-2009-02A/bin:$PATH"

echo ""
echo "Validating ShieldAgent imports."

python - <<'PY'
mods = [
    "agent",
    "prompt",
    "utils",
    "tools.aspm_tool",
    "tools.certification",
    "tools.verification",
    "tools.policy_extraction",
]
for m in mods:
    __import__(m)
    print("OK", m)
PY

python policy_extractor_async.py --help >/tmp/shieldagent_policy_extractor_help.txt
python agent/run_inference.py --help >/tmp/shieldagent_run_inference_help.txt 2>/tmp/shieldagent_run_inference_help.err || true

cd "$PCG_ROOT" || exit 1

python -m py_compile scripts/baselines/shieldagent/run_shieldagent_r1_r5.py

cat > "$PCG_ROOT/results/baselines/shieldagent/runtime/setup_env.sh" <<ENV
export PCG_ROOT="$PCG_ROOT"
export SHIELD_WORK_REPO="$SHIELD_WORK_REPO"
export SHIELD_VENV="$SHIELD_VENV"
export PYTHONPATH="$SHIELD_WORK_REPO:$SHIELD_WORK_REPO/agent:\${PYTHONPATH:-}"
export PATH="$SHIELD_WORK_REPO/agent/lib/LADR-2009-02A/bin:\$PATH"
ENV

echo ""
echo "SHIELDAGENT_SETUP_COMPLETE"
echo "Runtime venv: $SHIELD_VENV"
echo "Scratch author repo: $SHIELD_WORK_REPO"
echo ""
echo "To run ShieldAgent:"
echo "  source \"$SHIELD_VENV/bin/activate\""
echo "  bash scripts/baselines/shieldagent/run_shieldagent_interactive.sh"
