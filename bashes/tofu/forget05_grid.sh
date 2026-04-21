#!/usr/bin/env bash
#
# forget05 full grid sweep. Keeps A1 fixed at ep=5 (best so far).
# Phase 1: trains any missing A2 checkpoints (ep ∈ {2, 3, 4, 5}).
# Phase 2: runs the cartesian product of (A2 epoch) × (weight_a2) × (weight_a1).
#
# Default grid: 2 × 3 × 4 × 3 = 72  configs  →  ~18 hours with 15 min per eval.
# CHANGE the arrays below to fit your time budget.
#
# Usage:
#   bash bashes/tofu/forget05_grid.sh              # defaults below
#   GPU=0 A1_EP=5 bash bashes/tofu/forget05_grid.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
A1_EP="${A1_EP:-5}"
SPLIT="forget05_perturbed"
RSUB_PATH="data/rsub/forget05_k40.json"

# ============= GRID (edit to taste) =============
# NOTE: reducing any axis makes the run much shorter.
A2_EPOCHS=(2 3 4 5)                  # A2 training epochs
WEIGHTS_A1=(-0.6 -0.8 -1.0)          # A1 inference weight
WEIGHTS_A2=(0.2 0.5 0.8 1.0)          # A2 inference weight
FILTERS=(1e-2)                        # top_logit_filter (0.01 fixed; add 0.1 if wanted)
# ================================================

TOTAL=$(( ${#A2_EPOCHS[@]} * ${#WEIGHTS_A1[@]} * ${#WEIGHTS_A2[@]} * ${#FILTERS[@]} ))
echo "Total eval configs: $TOTAL  (each ~15min → ~$((TOTAL*15)) min total)"
echo

# A1 ckpt reused for all runs
A1_CKPT=$(find outputs_trained_models/tofu_dual_f05ep${A1_EP} -name "checkpoint-*" -type d -path "*a1*" \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -z "$A1_CKPT" ] && { echo "A1 ep=$A1_EP checkpoint not found. Retrain first."; exit 1; }
echo "A1 (fixed, ep=$A1_EP): $A1_CKPT"
echo

# ---------- Phase 1: make sure each A2 checkpoint exists ----------
ensure_a2() {
    local EP=$1
    local OUTDIR="outputs_trained_models/tofu_dual_f05_a2ep${EP}"
    local EXISTING
    EXISTING=$(find "$OUTDIR" -name "checkpoint-*" -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$EXISTING" ]; then
        echo "  A2 ep=$EP exists: $EXISTING"
        return
    fi
    echo "  Training A2 ep=$EP ..."
    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/hf_forget_train.py \
        project="grid_a2ep${EP}" \
        data=tofu \
        data.dataset.split=$SPLIT \
        data_mode=dual_a2 \
        data_mode.r_sub_indices_path=$RSUB_PATH \
        model=tofu-llama-2 \
        model_mode=uld \
        model_mode.num_layer=8 \
        unlearn_loss=remember+uniform \
        unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=8 \
        trainer.gradient_accumulation_steps=2 \
        trainer.learning_rate=1e-3 \
        trainer.max_epochs=$EP \
        trainer.strategy=gpu \
        OUTPUTMODELDIR=$OUTDIR \
        postfix="a2" \
        "hydra.run.dir=outputs/tune_log/grid_a2ep${EP}/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo "--- Phase 1: ensure A2 checkpoints exist ---"
for EP in "${A2_EPOCHS[@]}"; do
    ensure_a2 "$EP"
done
echo

# ---------- Phase 2: grid eval ----------
echo "--- Phase 2: grid eval ---"
RESULTS_FILE="/tmp/forget05_grid_results.csv"
echo "tag,A2_epoch,w1,w2,filter,FQ,MU,forget_proba" > "$RESULTS_FILE"

run_eval() {
    local A2_EP=$1 W1=$2 W2=$3 FILT=$4
    local TAG="a2e${A2_EP}_w1${W1//./p}_w2${W2//./p}_f${FILT//./p}"
    TAG="${TAG//-/m}"
    local A2_CKPT
    A2_CKPT=$(find "outputs_trained_models/tofu_dual_f05_a2ep${A2_EP}" -name "checkpoint-*" -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    [ -z "$A2_CKPT" ] && { echo "  [$TAG] A2 ckpt missing — skip"; return; }

    echo "  [$(date '+%H:%M:%S')] Running $TAG  (A2_ep=$A2_EP w1=$W1 w2=$W2 filter=$FILT)"

    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/eval_tofu.py \
        data=tofu \
        data.dataset.split=$SPLIT \
        data.dataset.eval.retain_result="data/forget05_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json" \
        data.dataset.eval.batch_size=4 \
        model=tofu-llama-2 \
        model_mode=dual_uld \
        model_mode.weight_a1=$W1 \
        model_mode.weight_a2=$W2 \
        model_mode.top_logit_filter=$FILT \
        model_mode.a1_ckpt_path="$A1_CKPT" \
        model_mode.a2_ckpt_path="$A2_CKPT" \
        ckpt_path="$A1_CKPT" \
        OUTDIRNAME="outputs_eval/f05_grid/$TAG" \
        "hydra.run.dir=outputs/tune_log/grid_${TAG}/\${now:%Y-%m-%d_%H-%M-%S}" \
        > /tmp/grid_${TAG}.log 2>&1

    local CSV
    CSV=$(find "outputs/tune_log/grid_${TAG}" -name "aggregate_stat.csv" 2>/dev/null | head -1)
    if [ -n "$CSV" ]; then
        local FQ MU FP
        FQ=$(awk -F',' 'NR==2 {printf "%.4f", $14}' "$CSV")
        MU=$(awk -F',' 'NR==2 {printf "%.4f", $13}' "$CSV")
        FP=$(awk -F',' 'NR==2 {printf "%.4f", $11}' "$CSV")
        echo "$TAG,$A2_EP,$W1,$W2,$FILT,$FQ,$MU,$FP" >> "$RESULTS_FILE"
        printf "    → FQ=%s  MU=%s  proba=%s\n" "$FQ" "$MU" "$FP"
    else
        echo "$TAG,$A2_EP,$W1,$W2,$FILT,,,ERROR" >> "$RESULTS_FILE"
        echo "    → NO CSV (eval failed)"
    fi
}

for A2_EP in "${A2_EPOCHS[@]}"; do
  for W1 in "${WEIGHTS_A1[@]}"; do
    for W2 in "${WEIGHTS_A2[@]}"; do
      for FILT in "${FILTERS[@]}"; do
        run_eval "$A2_EP" "$W1" "$W2" "$FILT"
      done
    done
  done
done

# ---------- Phase 3: summary ----------
echo
echo "============================================================"
echo "All done. Results in $RESULTS_FILE"
echo "Top 5 by FQ:"
echo "============================================================"
# sort by FQ desc, skip header and missing rows
awk -F',' 'NR>1 && $6!="" {print $6"\t"$7"\t"$1}' "$RESULTS_FILE" \
    | sort -rn | head -5 \
    | awk -F'\t' '{printf "  FQ=%s  MU=%s  cfg=%s\n",$1,$2,$3}'

echo
echo "Full table:"
column -s, -t "$RESULTS_FILE"
