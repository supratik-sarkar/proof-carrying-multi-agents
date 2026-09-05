#!/usr/bin/env bash
# Unattended local 4-headline-cell run at n=5, seed=0.
# Hits the 5 core experiments + 7 additional artifacts per cell.
# Wall-time estimate on M4 Pro: ~2-3 hours total.
set -uo pipefail   # NOT -e — keep going on partial failures so other cells finish

REPO="$HOME/Desktop/pcg-pcg-benchmark"
cd "$REPO"
source .venvs/multi-agents/bin/activate
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false

LOG_DIR="${PCG_ROOT}"
N=5
SEED=0
BACKEND=hf_local

# 4 headline cells we have weights cached for. Ordered by expected wall time
# (smallest model first, biggest last) so if the laptop wakes early you still
# have the cheap cells done.
CELLS=(
  "fever:phi-3.5-mini:microsoft/Phi-3.5-mini-instruct"
  "hotpotqa:qwen2.5-7B:Qwen/Qwen2.5-7B-Instruct"
  "pubmedqa:Llama-3.1-8B:meta-llama/Llama-3.1-8B-Instruct"
  "tatqa:Gemma-2-9b-it:google/gemma-2-9b-it"
)

CORE_EXPS=(
  "r1_checkability"
  "r2_redundancy"
  "r3_responsibility"
  "r4_risk_privacy"
  "r5_overhead"
)

date > "$LOG_DIR/STARTED.txt"

for CELL_SPEC in "${CELLS[@]}"; do
  IFS=':' read -r DS PAPER_MODEL HF_MODEL <<< "$CELL_SPEC"
  CELL_LABEL="${DS}:${PAPER_MODEL}"
  CELL_SAFE="${DS}__${PAPER_MODEL//\//_}"

  echo ""
  echo "======================================================================"
  echo " CELL: $CELL_LABEL"
  echo " hf_model=$HF_MODEL  n=$N  seed=$SEED  backend=$BACKEND"
  echo "======================================================================"
  echo "$(date): START $CELL_LABEL" >> "$LOG_DIR/timeline.log"

  # --- 5 core experiments ---
  for EXP in "${CORE_EXPS[@]}"; do
    LOG="$LOG_DIR/${CELL_SAFE}__core_${EXP}.log"
    EXTRA=""
    [ "$EXP" = "r2_redundancy" ] && EXTRA="--k-values 1 2 4"
    echo "  -> core $EXP  log=$LOG"
    {
      echo "=== $(date) core $EXP $CELL_LABEL ==="
      python scripts/experiments/run_${EXP}.py \
        --dataset "$DS" \
        --model "$HF_MODEL" \
        --backend "$BACKEND" \
        --n-examples "$N" \
        --seeds "$SEED" \
        $EXTRA
      echo "=== returncode $? ==="
    } > "$LOG" 2>&1 || echo "  WARN: core $EXP failed for $CELL_LABEL — see $LOG"
  done

  # --- 7-step additional-paper-artifacts pipeline ---
  LOG="$LOG_DIR/${CELL_SAFE}__additional_artifacts.log"
  echo "  -> additional artifacts  log=$LOG"
  {
    echo "=== $(date) additional artifacts $CELL_LABEL ==="
    bash scripts/runs/run_additional_paper_artifacts.sh \
      --cells "$CELL_LABEL" \
      --n-examples "$N" \
      --seed "$SEED" \
      --backend "$BACKEND"
    echo "=== returncode $? ==="
  } > "$LOG" 2>&1 || echo "  WARN: additional artifacts failed for $CELL_LABEL — see $LOG"

  echo "$(date): DONE $CELL_LABEL" >> "$LOG_DIR/timeline.log"
done

# --- Final build: regenerate aggregate tables/figures from EVERY JSON now on disk ---
echo ""
echo "======================================================================"
echo " FINAL BUILD: aggregating across all cells"
echo "======================================================================"
LOG="$LOG_DIR/__final_build.log"
{
  echo "=== $(date) final build ==="
  python scripts/common/build_additional_paper_artifacts.py
  echo "=== returncode $? ==="
} > "$LOG" 2>&1 || echo "  WARN: final build failed — see $LOG"

echo "$(date): ALL DONE" >> "$LOG_DIR/timeline.log"
date > "$LOG_DIR/FINISHED.txt"
