#!/usr/bin/env bash

# =====================================================================
# ShieldAgent / AutoPolicy interactive run for PCG-MAS
# =====================================================================
#
# Run from the PCG-MAS repo root after:
#
#   bash scripts/baselines/shieldagent/setup_shieldagent.sh
#
# This script:
#   1. activates the ShieldAgent venv
#   2. asks for OpenAI / Anthropic keys
#   3. extracts or reuses a ShieldAgent policy bank
#   4. asks which PCG-MAS cells/seeds/n_examples to evaluate
#   5. runs ShieldAgent R1-R5 on matching PCG-MAS baseline_inputs
#   6. validates generated artifacts
#   7. optionally switches back to the PCG-MAS venv
# =====================================================================

echo "ShieldAgent / AutoPolicy interactive baseline runner"

PCG_ROOT="$(pwd)"

printf "ShieldAgent virtual environment name [default: shield-agent]: "
read -r SHIELD_VENV_NAME
SHIELD_VENV_NAME="${SHIELD_VENV_NAME:-shield-agent}"

printf "PCG-MAS virtual environment name to return to later [default: multi-agents]: "
read -r PCG_VENV_NAME
PCG_VENV_NAME="${PCG_VENV_NAME:-multi-agents}"

SHIELD_VENV="$PCG_ROOT/.venvs/$SHIELD_VENV_NAME"
PCG_VENV="$PCG_ROOT/$PCG_VENV_NAME"
SHIELD_WORK_REPO="$PCG_ROOT/.sota_src/shieldagent_author"

if [[ ! -d "$SHIELD_VENV" ]]; then
  echo "ShieldAgent venv not found: $SHIELD_VENV"
  echo "Run setup first:"
  echo "  bash scripts/baselines/shieldagent/setup_shieldagent.sh"
  exit 1
fi

if [[ ! -d "$SHIELD_WORK_REPO" ]]; then
  echo "ShieldAgent scratch repo not found: $SHIELD_WORK_REPO"
  echo "Run setup first:"
  echo "  bash scripts/baselines/shieldagent/setup_shieldagent.sh"
  exit 1
fi

deactivate 2>/dev/null || true
source "$SHIELD_VENV/bin/activate"

echo ""
echo "Using ShieldAgent Python:"
which python
python --version

export PCG_ROOT
export SHIELD_WORK_REPO
export SHIELD_VENV
export PYTHONPATH="$SHIELD_WORK_REPO:$SHIELD_WORK_REPO/agent:$PCG_ROOT:${PYTHONPATH:-}"
export PATH="$SHIELD_WORK_REPO/agent/lib/LADR-2009-02A/bin:$PATH"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-5}"

echo ""
echo "API keys."
printf "OPENAI_API_KEY [hidden; required for policy extraction with gpt-4o]: "
read -rs OPENAI_API_KEY
printf "\nANTHROPIC_API_KEY [hidden; required for ShieldAgent evaluation]: "
read -rs ANTHROPIC_API_KEY
printf "\nHF_TOKEN [hidden; optional]: "
read -rs HF_TOKEN
printf "\n"

export OPENAI_API_KEY
export ANTHROPIC_API_KEY
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is empty. Policy extraction may fail unless reusing an existing selected policy bank."
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is required for ShieldAgent R1-R5 evaluation."
  exit 1
fi

cd "$PCG_ROOT" || exit 1

mkdir -p results/baselines/shieldagent/raw
mkdir -p results/baselines/shieldagent/selected_policy_bank
mkdir -p results/baselines/shieldagent/r1_r5
mkdir -p results/tables/csv/shieldagent_outputs

echo ""
echo "Policy bank options:"
echo "  1. Reuse existing selected policy bank if present"
echo "  2. Extract from bundled EU AI Act Article 5 PDF"
echo "  3. Extract from a custom policy document path or URL"
printf "Choose policy option [default: 1]: "
read -r POLICY_OPTION
POLICY_OPTION="${POLICY_OPTION:-1}"

BANK_DIR="$PCG_ROOT/results/baselines/shieldagent/selected_policy_bank"

need_extract="n"

if [[ "$POLICY_OPTION" == "1" ]]; then
  if [[ -f "$BANK_DIR/policies.json" && -f "$BANK_DIR/rules.json" && -f "$BANK_DIR/risk_categories.json" ]]; then
    echo "Reusing existing selected policy bank:"
    find "$BANK_DIR" -maxdepth 1 -type f | sort
  else
    echo "Existing selected policy bank not found. Falling back to bundled EU AI Act Article 5 extraction."
    need_extract="y"
    POLICY_OPTION="2"
  fi
else
  need_extract="y"
fi

if [[ "$POLICY_OPTION" == "2" ]]; then
  POLICY_DOC="$SHIELD_WORK_REPO/policy_docs/eu_ai_act_art5.pdf"
  POLICY_TYPE="pdf"
  POLICY_ORG="EU AI Act Article 5"
  POLICY_RANGE="1-8"
  need_extract="y"
elif [[ "$POLICY_OPTION" == "3" ]]; then
  printf "Policy document path or URL: "
  read -r POLICY_DOC

  printf "Policy document input type pdf/html/txt [default: pdf]: "
  read -r POLICY_TYPE
  POLICY_TYPE="${POLICY_TYPE:-pdf}"

  printf "Policy organization/name [default: Custom Policy]: "
  read -r POLICY_ORG
  POLICY_ORG="${POLICY_ORG:-Custom Policy}"

  printf "Initial page range for PDF [default: 1-8]: "
  read -r POLICY_RANGE
  POLICY_RANGE="${POLICY_RANGE:-1-8}"

  need_extract="y"
fi

if [[ "$need_extract" == "y" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Cannot extract policies without OPENAI_API_KEY."
    exit 1
  fi

  SLUG="$(python - <<PY
import re
s = """$POLICY_ORG""".strip().lower()
s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
print(s or "policy")
PY
)"

  RAW_OUT="$PCG_ROOT/results/baselines/shieldagent/raw/$SLUG"
  rm -rf "$RAW_OUT"
  mkdir -p "$RAW_OUT"

  cd "$SHIELD_WORK_REPO" || exit 1

  echo ""
  echo "Extracting ShieldAgent policy bank."
  echo "POLICY_DOC=$POLICY_DOC"
  echo "POLICY_TYPE=$POLICY_TYPE"
  echo "POLICY_ORG=$POLICY_ORG"
  echo "POLICY_RANGE=$POLICY_RANGE"

  if [[ "$POLICY_TYPE" == "pdf" ]]; then
    python policy_extractor_async.py \
      -d "$POLICY_DOC" \
      -t "$POLICY_TYPE" \
      -org "$POLICY_ORG" \
      -ipr "$POLICY_RANGE" \
      -o "$RAW_OUT" \
      -m gpt-4o \
      -er \
      --debug
  else
    python policy_extractor_async.py \
      -d "$POLICY_DOC" \
      -t "$POLICY_TYPE" \
      -org "$POLICY_ORG" \
      -o "$RAW_OUT" \
      -m gpt-4o \
      -er \
      --debug
  fi

  cd "$PCG_ROOT" || exit 1

  echo ""
  echo "Selecting latest nonzero extracted policy bank."

  python - <<'PY'
import json
import shutil
from pathlib import Path

root = Path("results/baselines/shieldagent/raw")
bank = Path("results/baselines/shieldagent/selected_policy_bank")
bank.mkdir(parents=True, exist_ok=True)

reports = sorted(root.glob("**/*_extraction_report.json"))
candidates = []

for report in reports:
    try:
        obj = json.loads(report.read_text())
    except Exception:
        continue
    n = int(obj.get("policies_extracted") or 0)
    if n <= 0:
        continue

    d = report.parent
    policies = sorted(d.glob("*_all_extracted_policies.json"))
    rules = sorted(d.glob("*_all_extracted_rules.json"))
    risk = sorted(d.glob("*_risk_categories.json"))
    mapping = sorted(d.glob("*_policy_rule_mapping.json"))

    if policies and rules and risk:
        candidates.append({
            "report": report,
            "dir": d,
            "n": n,
            "policies": policies[-1],
            "rules": rules[-1],
            "risk": risk[-1],
            "mapping": mapping[-1] if mapping else None,
        })

if not candidates:
    raise SystemExit("No nonzero ShieldAgent policy extraction found. Try a richer policy document or broader page range.")

chosen = candidates[-1]

for dst_name, src in [
    ("policies.json", chosen["policies"]),
    ("rules.json", chosen["rules"]),
    ("risk_categories.json", chosen["risk"]),
    ("extraction_report.json", chosen["report"]),
]:
    shutil.copy2(src, bank / dst_name)

if chosen["mapping"]:
    shutil.copy2(chosen["mapping"], bank / "policy_rule_mapping.json")

summary = {
    "selected_policy_source_dir": str(chosen["dir"]),
    "policies_extracted_reported": chosen["n"],
    "policy_bank_files": {
        "policies": str(bank / "policies.json"),
        "rules": str(bank / "rules.json"),
        "risk_categories": str(bank / "risk_categories.json"),
        "policy_rule_mapping": str(bank / "policy_rule_mapping.json"),
        "extraction_report": str(bank / "extraction_report.json"),
    },
}
(bank / "selected_policy_bank_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
fi

cd "$PCG_ROOT" || exit 1

echo ""
echo "Choose ShieldAgent benchmark cells using dataset:model syntax."
echo "Examples:"
echo "  fever:phi-3.5-mini,tatqa:phi-3.5-mini"
echo "  fever:phi-3.5-mini,hotpotqa:qwen2.5-7b"
echo "  pubmedqa:llama-3.1-8b,tatqa:gemma-2-9b-it"
echo "  all"
echo ""
echo "Use 'all' to infer available datasets from results/tables/csv/baseline_inputs."
printf "Cells to run [default: fever:phi-3.5-mini,tatqa:phi-3.5-mini]: "
read -r CELLS
CELLS="${CELLS:-fever:phi-3.5-mini,tatqa:phi-3.5-mini}"


printf "Seeds, comma-separated [default: 0]: "
read -r SEEDS
SEEDS="${SEEDS:-0}"

printf "Number of examples per dataset/seed [default: 3]: "
read -r N_EXAMPLES
N_EXAMPLES="${N_EXAMPLES:-3}"

echo ""
echo "Previewing resolved ShieldAgent cells."

python scripts/baselines/shieldagent/run_shieldagent_r1_r5.py \
  --pairs "$CELLS" \
  --seeds "$SEEDS" \
  --n-examples "$N_EXAMPLES" \
  --list-cells

echo ""
printf "Proceed with ShieldAgent R1-R5 run? y/n [default: y]: "
read -r PROCEED
PROCEED="${PROCEED:-y}"

if [[ "$PROCEED" != "y" && "$PROCEED" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

echo ""
echo "Running ShieldAgent R1-R5 comparative evaluation."

python scripts/baselines/shieldagent/run_shieldagent_r1_r5.py \
  --pairs "$CELLS" \
  --seeds "$SEEDS" \
  --n-examples "$N_EXAMPLES"

echo ""
echo "Validating generated ShieldAgent artifacts."

python - <<'PY'
import json
from pathlib import Path

root = Path("results/baselines/shieldagent/r1_r5")

manifest_path = root / "manifest.json"
aggregate_path = root / "aggregate_by_dataset_model.json"

if not manifest_path.exists():
    raise SystemExit("Missing manifest.json")

if not aggregate_path.exists():
    raise SystemExit("Missing aggregate_by_dataset_model.json")

manifest = json.loads(manifest_path.read_text())

summary_files = [Path(p) for p in manifest.get("summary_files", [])]
if not summary_files:
    raise SystemExit("No summary files recorded in manifest.")

required_leafs = [
    "input.jsonl",
    "r1_checkability.jsonl",
    "r2_redundancy.jsonl",
    "r3_responsibility.jsonl",
    "r4_risk_control.json",
    "r5_overhead.json",
    "summary.json",
]

missing = []

for summary_path in summary_files:
    cell_dir = summary_path.parent
    for leaf in required_leafs:
        p = cell_dir / leaf
        if not p.exists():
            missing.append(str(p))

if missing:
    raise SystemExit("Missing ShieldAgent artifacts:\n" + "\n".join(missing))

print("Per-cell summaries:")
for summary_path in summary_files:
    obj = json.loads(summary_path.read_text())
    print(json.dumps({
        "summary": str(summary_path),
        "dataset": obj["dataset"],
        "model": obj["model"],
        "seed": obj["seed"],
        "n": obj["n"],
        "R1_accept_rate": obj["R1_checkability"]["accept_rate"],
        "R1_block_rate": obj["R1_checkability"]["block_rate"],
        "R1_verify_rate": obj["R1_checkability"]["verify_rate"],
        "R1_false_accept_proxy_rate": obj["R1_checkability"]["false_accept_proxy_rate_among_known"],
        "R2_quorum_accept_n": obj["R2_redundancy"]["quorum_accept_n"],
        "R2_quorum_block_n": obj["R2_redundancy"]["quorum_block_n"],
        "R2_quorum_verify_n": obj["R2_redundancy"]["quorum_verify_n"],
        "R3_total_decision_flips": obj["R3_responsibility"]["total_decision_flips"],
        "R5_latency_mean_s": obj["R5_overhead"]["latency_mean_s"],
        "R5_tokens_est_total": obj["R5_overhead"]["tokens_est_total"],
    }, indent=2, sort_keys=True))

print("Aggregate by dataset:model:")
agg = json.loads(aggregate_path.read_text())
print(json.dumps(agg, indent=2, sort_keys=True))

print("SHIELDAGENT_REPRODUCIBILITY_RUN_VERIFIED")
PY

echo ""
echo "Generated ShieldAgent R1-R5 files:"
find results/baselines/shieldagent/r1_r5 -maxdepth 3 -type f | sort

echo ""
printf "Switch back to PCG-MAS venv '$PCG_VENV_NAME'? y/n [default: y]: "
read -r SWITCH_BACK
SWITCH_BACK="${SWITCH_BACK:-y}"

if [[ "$SWITCH_BACK" == "y" || "$SWITCH_BACK" == "Y" ]]; then
  deactivate 2>/dev/null || true

  if [[ -d "$PCG_VENV" ]]; then
    source "$PCG_VENV/bin/activate"
    export PYTHONPATH="$PCG_ROOT:${PYTHONPATH:-}"
    echo "Switched back to PCG-MAS venv:"
    which python
    python --version
  else
    echo "PCG-MAS venv not found: $PCG_VENV"
    echo "Activate it manually if needed."
  fi
fi

echo ""
echo "SHIELDAGENT_INTERACTIVE_RUN_COMPLETE"
