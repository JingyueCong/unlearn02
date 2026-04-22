#!/usr/bin/env bash
#
# FLAT baseline (Wang et al. 2024, arXiv:2410.11143) on TOFU.
# Forget-data-only f-divergence variational unlearning.
# Runs train + eval on forget01, forget05, forget10 — outputs FQ, MU, R-L.
#
# Method: max f-divergence between template "I don't know"-style answers
# (from data/refusal.jsonl) and ground-truth forget answers on forget queries.
# No retain data used.
#
# LoRA on top of full 7B base (num_layer=0). f-div = JS (default).
#
# Total cost: ~3 splits × (~8 min train + ~20 min eval) ≈ 1.5 hours.
#
# Usage:
#   bash bashes/tofu/flat_baseline.sh
#   GPU=0 SPLITS="forget05 forget10" bash bashes/tofu/flat_baseline.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SPLITS="${SPLITS:-forget01 forget05 forget10}"
OUTPUTMODELDIR="${OUTPUTMODELDIR:-outputs_trained_models/tofu_flat}"
RESULTS_FILE="/tmp/flat_baseline_results.csv"

echo "split,FQ,MU,Forget_ROUGE,Retain_ROUGE,Real_Authors_ROUGE,Real_World_ROUGE,Forget_Proba,Retain_Proba" > "$RESULTS_FILE"

run_split() {
    local SPLIT=$1
    local TAG="flat_${SPLIT}"
    local TRAIN_OUT="${OUTPUTMODELDIR}/${TAG}"
    local BASELINE_PATH="data/${SPLIT}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"

    if [ ! -f "$BASELINE_PATH" ]; then
        echo "MISSING retain baseline at $BASELINE_PATH — skip $SPLIT"
        echo "$SPLIT,,,,,,,," >> "$RESULTS_FILE"
        return
    fi

    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] $SPLIT - Phase 1: training FLAT"
    echo "============================================================"

    local EXISTING
    EXISTING=$(find "$TRAIN_OUT" -name "checkpoint-*" -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$EXISTING" ]; then
        CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/hf_forget_train.py \
            project="$TAG" \
            data=tofu \
            data.dataset.split=${SPLIT}_perturbed \
            data_mode=dpo \
            model=tofu-llama-2 \
            model_mode=uld \
            model_mode.num_layer=0 \
            unlearn_loss=flat \
            trainer.batch_size=4 \
            trainer.gradient_accumulation_steps=4 \
            trainer.learning_rate=2e-5 \
            trainer.max_epochs=5 \
            trainer.strategy=gpu \
            OUTPUTMODELDIR="$TRAIN_OUT" \
            postfix="flat" \
            "hydra.run.dir=outputs/tune_log/${TAG}/\${now:%Y-%m-%d_%H-%M-%S}"

        EXISTING=$(find "$TRAIN_OUT" -name "checkpoint-*" -type d \
            | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    [ -z "$EXISTING" ] && { echo "ERROR: no ckpt for $SPLIT"; echo "$SPLIT,,,,,,,," >> "$RESULTS_FILE"; return; }
    echo "  ckpt: $EXISTING"

    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] $SPLIT - Phase 2: evaluating"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/eval_tofu.py \
        data=tofu \
        data.dataset.split=${SPLIT}_perturbed \
        data.dataset.eval.retain_result="$BASELINE_PATH" \
        data.dataset.eval.batch_size=4 \
        model=tofu-llama-2 \
        model_mode=base \
        ckpt_path="$EXISTING" \
        OUTDIRNAME="outputs_eval/flat_baseline/$SPLIT" \
        "hydra.run.dir=outputs/tune_log/eval_${TAG}/\${now:%Y-%m-%d_%H-%M-%S}"

    local CSV
    CSV=$(find "outputs/tune_log/eval_${TAG}" -name "aggregate_stat.csv" | head -1)
    if [ -n "$CSV" ]; then
        awk -F',' -v S="$SPLIT" 'NR==2 {
            printf "%s,%.4g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   S, $14, $13, $10, $1, $4, $7, $11, $2
        }' "$CSV" >> "$RESULTS_FILE"
        awk -F',' 'NR==2 { printf "  → FQ=%.4g  MU=%.4f  Forget-RL=%.4f  Retain-RL=%.4f\n", $14, $13, $10, $1 }' "$CSV"
    else
        echo "$SPLIT,,,,,,,," >> "$RESULTS_FILE"
        echo "  → (no CSV)"
    fi
}

for SPLIT in $SPLITS; do
    run_split "$SPLIT"
done

echo
echo "============================================================"
echo "FLAT baseline — final table"
echo "============================================================"
column -s, -t "$RESULTS_FILE"
