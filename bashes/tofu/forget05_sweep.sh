#!/usr/bin/env bash
#
# Sequential weight/filter sweep on forget05 using the existing ep=5 checkpoints.
# Reuses A1+A2 trained at ep=5 (best so far); only varies inference params.
#
# Configurations:
#   1) weight_a1=-0.6, weight_a2=0.6   (symmetric, gentler)
#   2) weight_a1=-0.8, weight_a2=0.8, top_logit_filter=0.1   (wider filter)
#   3) weight_a1=-0.8, weight_a2=0.0   (disable A2, diagnostic)
#
# Each eval ~15min. Total ~45min.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SPLIT="forget05_perturbed"

A1_CKPT=$(find outputs_trained_models/tofu_dual_f05ep5 -name "checkpoint-*" -type d -path "*ep5_a1*" \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_CKPT=$(find outputs_trained_models/tofu_dual_f05ep5 -name "checkpoint-*" -type d -path "*ep5_a2*" \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

echo "A1=$A1_CKPT"
echo "A2=$A2_CKPT"
[ -z "$A1_CKPT" ] && { echo "A1 ckpt not found"; exit 1; }
[ -z "$A2_CKPT" ] && { echo "A2 ckpt not found"; exit 1; }

run_eval() {
    local TAG="$1" W1="$2" W2="$3" FILTER="$4"
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] Running $TAG: w1=$W1 w2=$W2 filter=$FILTER"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/eval_tofu.py \
        data=tofu \
        data.dataset.split=$SPLIT \
        data.dataset.eval.retain_result="data/forget05_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json" \
        data.dataset.eval.batch_size=4 \
        model=tofu-llama-2 \
        model_mode=dual_uld \
        model_mode.weight_a1=$W1 \
        model_mode.weight_a2=$W2 \
        model_mode.top_logit_filter=$FILTER \
        model_mode.a1_ckpt_path="$A1_CKPT" \
        model_mode.a2_ckpt_path="$A2_CKPT" \
        ckpt_path="$A1_CKPT" \
        OUTDIRNAME=outputs_eval/tofu-dual-uld-f05sweep/$TAG \
        "hydra.run.dir=outputs/tune_log/dual_f05sweep_${TAG}/\${now:%Y-%m-%d_%H-%M-%S}"
}

# 1) symmetric gentler
run_eval "w06"        -0.6 0.6 1e-2

# 2) wider filter
run_eval "filter01"   -0.8 0.8 0.1

# 3) disable A2 (diagnostic)
run_eval "no_a2"      -0.8 0.0 1e-2

# ----- summary -----
echo
echo "============================================================"
echo "All sweeps complete. Summary (Forget Quality / Model Utility):"
echo "============================================================"
for TAG in w06 filter01 no_a2; do
    CSV=$(find outputs/tune_log/dual_f05sweep_${TAG} -name "aggregate_stat.csv" 2>/dev/null | head -1)
    if [ -n "$CSV" ]; then
        FQ=$(awk -F',' 'NR==2 {print $14}' "$CSV")
        MU=$(awk -F',' 'NR==2 {print $13}' "$CSV")
        printf "  %-12s  FQ=%-10s  MU=%-10s\n" "$TAG" "$FQ" "$MU"
    else
        printf "  %-12s  (no CSV found)\n" "$TAG"
    fi
done
