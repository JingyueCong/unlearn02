#!/usr/bin/env bash
#
# Dual-ULD end-to-end pipeline for TOFU:
#   1. Select R_sub (retain items most similar to forget items) via embedding sim.
#   2. Train A1 (small assistant) on forget+R_sub (GD) + R_far+perturb (Uniform).
#   3. Train A2 (separate small assistant) on R_sub (GD) + rest (Uniform).
#   4. Evaluate with final = base - A1 + A2.

set -e

# Resolve repo root (parent of bashes/tofu/) and prepend to PYTHONPATH so that
# `import uld` picks up THIS copy, not any other pip-installed editable copy.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

master_port=43244

# GPUs to use. Override by exporting before invoking, e.g.:
#   GPUS=0,1 bash bashes/tofu/dual_uld_pipeline.sh
# `EVAL_GPU` is the single device used for evaluation (defaults to first of GPUS).
GPUS="${GPUS:-0}"
NPROC=$(echo "$GPUS" | awk -F',' '{print NF}')
EVAL_GPU="${EVAL_GPU:-${GPUS%%,*}}"

split="${SPLIT:-forget10}"             # forget01 | forget05 | forget10
K="${K:-80}"                           # R_sub size (tune: ~ |forget| * 1~2x)
OUTPUTMODELDIR="${OUTPUTMODELDIR:-outputs_trained_models/tofu_dual}"
EVAL_OUTDIRNAME="${EVAL_OUTDIRNAME:-outputs_eval/tofu-dual-uld}"
RSUB_PATH="data/rsub/${split}_k${K}.json"

echo "GPUS=$GPUS  NPROC=$NPROC  EVAL_GPU=$EVAL_GPU  split=$split  K=$K"

lr=1e-3
# Use ddp when >=2 GPUs available; otherwise single-device mode (no torchrun).
if [ "$NPROC" -gt 1 ]; then
    STRATEGY="ddp"
else
    STRATEGY="gpu"
fi
COMMON_TRAIN="data=tofu trainer.batch_size=8 trainer.learning_rate=$lr \
    trainer.gradient_accumulation_steps=2 model=tofu-llama-2 \
    model_mode=uld model_mode.num_layer=8 trainer.strategy=$STRATEGY \
    unlearn_loss=remember+uniform OUTPUTMODELDIR=$OUTPUTMODELDIR"

# launcher: torchrun for multi-GPU, plain python for single-GPU.
launch_train() {
    local mport="$1"; shift
    if [ "$NPROC" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NPROC" --master_port="$mport" "$@"
    else
        CUDA_VISIBLE_DEVICES="$GPUS" python "$@"
    fi
}

# ---------- Step 1: select R_sub ----------
if [ ! -f "$RSUB_PATH" ]; then
    echo "[1/4] Selecting R_sub -> $RSUB_PATH"
    python scripts/select_rsub.py \
        --forget_split ${split}_perturbed \
        --k $K \
        --out $RSUB_PATH
else
    echo "[1/4] R_sub file already exists: $RSUB_PATH"
fi

# ---------- Step 2: train A1 ----------
echo "[2/4] Training A1"
launch_train $master_port \
    scripts/hf_forget_train.py \
    project="dual_uld_a1" \
    data.dataset.split=${split}_perturbed \
    data_mode=dual_a1 \
    data_mode.r_sub_indices_path=$RSUB_PATH \
    $COMMON_TRAIN \
    postfix="a1" \
    'hydra.run.dir=outputs/tune_log/dual_uld_a1/${now:%Y-%m-%d_%H-%M-%S}'

# ---------- Step 3: train A2 ----------
echo "[3/4] Training A2"
launch_train $((master_port+1)) \
    scripts/hf_forget_train.py \
    project="dual_uld_a2" \
    data.dataset.split=${split}_perturbed \
    data_mode=dual_a2 \
    data_mode.r_sub_indices_path=$RSUB_PATH \
    $COMMON_TRAIN \
    postfix="a2" \
    'hydra.run.dir=outputs/tune_log/dual_uld_a2/${now:%Y-%m-%d_%H-%M-%S}'

# ---------- Step 4: evaluate ----------
# Pick the latest checkpoint of each. Adjust grep if needed.
A1_CKPT=$(find $OUTPUTMODELDIR -type d -path "*a1*" -name "checkpoint-*" \
    | sort -t- -k2n | tail -n1)
A2_CKPT=$(find $OUTPUTMODELDIR -type d -path "*a2*" -name "checkpoint-*" \
    | sort -t- -k2n | tail -n1)

echo "[4/4] Evaluating"
echo "  A1 = $A1_CKPT"
echo "  A2 = $A2_CKPT"

CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_tofu.py \
    data=tofu \
    data.dataset.split=${split}_perturbed \
    data.dataset.eval.retain_result="data/${split}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json" \
    data.dataset.eval.batch_size=4 \
    model=tofu-llama-2 \
    model_mode=dual_uld \
    model_mode.weight_a1=-0.8 \
    model_mode.weight_a2=0.8 \
    model_mode.top_logit_filter=1e-2 \
    model_mode.a1_ckpt_path="$A1_CKPT" \
    model_mode.a2_ckpt_path="$A2_CKPT" \
    ckpt_path="$A1_CKPT" \
    OUTDIRNAME=${EVAL_OUTDIRNAME}/${split} \
    'hydra.run.dir=outputs/tune_log/dual_uld_eval/${now:%Y-%m-%d_%H-%M-%S}'
