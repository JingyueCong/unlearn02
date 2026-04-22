#!/usr/bin/env bash
#
# Unilogit baseline (arXiv:2505.06027, Vasilev et al. 2025) on TOFU.
# Runs train + eval on forget01, forget05, forget10 — outputs FQ, MU, R-L.
#
# Method: reverse-KL distillation with modified target logits so target prob
# becomes 1/|V|. No assistant model. Applied via LoRA on top of the full 7B
# base (num_layer=0 means keep all 32 layers).
#
# Total cost: 3 splits × (~15 min train + ~20 min eval) ≈ 2 hours per GPU.
#
# Usage:
#   bash bashes/tofu/unilogit_baseline.sh
#   GPU=0 SPLITS="forget05 forget10" bash bashes/tofu/unilogit_baseline.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SPLITS="${SPLITS:-forget01 forget05 forget10}"
OUTPUTMODELDIR="${OUTPUTMODELDIR:-outputs_trained_models/tofu_unilogit}"
RESULTS_FILE="/tmp/unilogit_baseline_results.csv"

# CSV header: split, FQ, MU, Forget_ROUGE, Retain_ROUGE, Real_Authors_ROUGE, Real_World_ROUGE, Forget_Proba, Retain_Proba
echo "split,FQ,MU,Forget_ROUGE,Retain_ROUGE,Real_Authors_ROUGE,Real_World_ROUGE,Forget_Proba,Retain_Proba" > "$RESULTS_FILE"

run_split() {
    local SPLIT=$1
    local TAG="unilogit_${SPLIT}"
    local TRAIN_OUT="${OUTPUTMODELDIR}/${TAG}"
    local BASELINE_PATH="data/${SPLIT}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"

    if [ ! -f "$BASELINE_PATH" ]; then
        echo "MISSING retain baseline at $BASELINE_PATH — skip $SPLIT"
        echo "$SPLIT,,,,,,,," >> "$RESULTS_FILE"
        return
    fi

    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] $SPLIT - Phase 1: training Unilogit (LoRA on full 7B)"
    echo "============================================================"

    # Train only if no ckpt exists
    local EXISTING_CKPT
    EXISTING_CKPT=$(find "$TRAIN_OUT" -name "checkpoint-*" -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$EXISTING_CKPT" ]; then
        CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/hf_forget_train.py \
            project="$TAG" \
            data=tofu \
            data.dataset.split=${SPLIT}_perturbed \
            data_mode=forget_more_retain_perturb \
            model=tofu-llama-2 \
            model_mode=uld \
            model_mode.num_layer=0 \
            unlearn_loss=unilogit+gd \
            trainer.batch_size=8 \
            trainer.gradient_accumulation_steps=2 \
            trainer.learning_rate=1e-4 \
            trainer.max_epochs=10 \
            trainer.strategy=gpu \
            OUTPUTMODELDIR="$TRAIN_OUT" \
            postfix="unilogit" \
            "hydra.run.dir=outputs/tune_log/${TAG}/\${now:%Y-%m-%d_%H-%M-%S}"

        EXISTING_CKPT=$(find "$TRAIN_OUT" -name "checkpoint-*" -type d \
            | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    [ -z "$EXISTING_CKPT" ] && { echo "ERROR: no ckpt produced for $SPLIT"; echo "$SPLIT,,,,,,,," >> "$RESULTS_FILE"; return; }
    echo "  ckpt: $EXISTING_CKPT"

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
        ckpt_path="$EXISTING_CKPT" \
        OUTDIRNAME="outputs_eval/unilogit_baseline/$SPLIT" \
        "hydra.run.dir=outputs/tune_log/eval_${TAG}/\${now:%Y-%m-%d_%H-%M-%S}"

    local CSV
    CSV=$(find "outputs/tune_log/eval_${TAG}" -name "aggregate_stat.csv" | head -1)
    if [ -n "$CSV" ]; then
        # aggregate_stat.csv columns (from tofu_eval):
        # 1:Retain_ROUGE  2:Retain_Probability  3:Retain_Truth_Ratio
        # 4:Real_Authors_ROUGE  5:Real_Authors_Probability  6:Real_Authors_Truth_Ratio
        # 7:Real_World_ROUGE  8:Real_World_Probability  9:Real_World_Truth_Ratio
        # 10:Forget_ROUGE  11:Forget_Probability  12:Forget_Truth_Ratio
        # 13:Model_Utility  14:Forget_Quality
        local FQ MU FR RR RAR RWR FP RP
        FQ=$(awk -F',' 'NR==2 {printf "%.4f", $14}' "$CSV")
        MU=$(awk -F',' 'NR==2 {printf "%.4f", $13}' "$CSV")
        FR=$(awk -F',' 'NR==2 {printf "%.4f", $10}' "$CSV")
        RR=$(awk -F',' 'NR==2 {printf "%.4f", $1}' "$CSV")
        RAR=$(awk -F',' 'NR==2 {printf "%.4f", $4}' "$CSV")
        RWR=$(awk -F',' 'NR==2 {printf "%.4f", $7}' "$CSV")
        FP=$(awk -F',' 'NR==2 {printf "%.4f", $11}' "$CSV")
        RP=$(awk -F',' 'NR==2 {printf "%.4f", $2}' "$CSV")
        echo "$SPLIT,$FQ,$MU,$FR,$RR,$RAR,$RWR,$FP,$RP" >> "$RESULTS_FILE"
        printf "  → FQ=%s  MU=%s  Forget-RL=%s  Retain-RL=%s\n" "$FQ" "$MU" "$FR" "$RR"
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
echo "Unilogit baseline — final table"
echo "============================================================"
column -s, -t "$RESULTS_FILE"
