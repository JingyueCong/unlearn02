#!/usr/bin/env bash
#
# Fine-grained checkpoint scan around the FQ peak found at step=700.
#
# Retrains A1 with save_steps=5 (200 ckpts for 1000 steps → ~14 GB disk).
# Then evals:
#   - Fine steps near peak 700:    675 685 690 695 700 705 710 715
#   - Filling gap step 800→1000:   820 840 860 880 900 920 940 960 980 1000
#
# This finds the EXACT peak at 5-step granularity and checks if there's a
# higher peak in the previously-missing step 800-1000 region.
#
# Total cost:
#   Phase 1 (A1 retrain save_steps=5):  ~6 min
#   Phase 2 (18 evals × ~12 min):       ~3.6 hours
#
# Usage:
#   bash bashes/tofu/forget05_fine_scan.sh
#   GPU=0 bash bashes/tofu/forget05_fine_scan.sh
#   CKPT_STEPS="..." bash ...   # custom list

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SPLIT="forget05_perturbed"
RSUB_PATH="data/rsub/forget05_k40.json"

# Fine-grained step list: near peak 700 + fill 800→1000 gap
CKPT_STEPS="${CKPT_STEPS:-675 685 690 695 700 705 710 715 725 775 820 840 860 880 900 920 940 960 980 1000}"

# ---------- Phase 0: ensure A2 (ep=3 K=40) exists ----------
A2_OUTDIR="outputs_trained_models/tofu_dual_f05a2ep3"
A2_CKPT=$(find "$A2_OUTDIR" -name "checkpoint-*" -type d 2>/dev/null \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -z "$A2_CKPT" ]; then
    echo "============================================================"
    echo "Phase 0: training A2 (ep=3, K=40, rw=5)"
    echo "============================================================"
    rm -rf "$A2_OUTDIR"
    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/hf_forget_train.py \
        project="dual_f05a2ep3" \
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
        trainer.max_epochs=3 \
        trainer.strategy=gpu \
        OUTPUTMODELDIR="$A2_OUTDIR" \
        postfix="a2" \
        "hydra.run.dir=outputs/tune_log/dual_f05a2ep3/\${now:%Y-%m-%d_%H-%M-%S}"
    A2_CKPT=$(find "$A2_OUTDIR" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fi
echo "A2: $A2_CKPT"

# ---------- Phase 1: A1 fine-grained training (save_steps=5) ----------
A1_OUTDIR="outputs_trained_models/tofu_dual_f05_a1finerun"

# Check if fine-run already exists with enough coverage (has ckpt-695 which only save_steps=5 produces)
EXISTING_FINE=$(find "$A1_OUTDIR" -name "checkpoint-695" -type d 2>/dev/null | head -1)
if [ -n "$EXISTING_FINE" ]; then
    A1_RUNDIR=$(dirname "$EXISTING_FINE")
    echo "Phase 1: A1 fine-run already exists at $A1_RUNDIR — skipping"
else
    echo "============================================================"
    echo "Phase 1: training A1 max_epochs=10 save_steps=5 (fresh ~14GB)"
    echo "============================================================"
    rm -rf "$A1_OUTDIR"

    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/hf_forget_train.py \
        project="dual_f05_a1finerun" \
        data=tofu \
        data.dataset.split=$SPLIT \
        data_mode=dual_a1 \
        data_mode.r_sub_indices_path=$RSUB_PATH \
        model=tofu-llama-2 \
        model_mode=uld \
        model_mode.num_layer=8 \
        unlearn_loss=remember+uniform \
        unlearn_loss.retain_weight=2.0 \
        trainer.batch_size=8 \
        trainer.gradient_accumulation_steps=2 \
        trainer.learning_rate=1e-3 \
        trainer.max_epochs=10 \
        trainer.strategy=gpu \
        +trainer.save_steps=5 \
        +trainer.eval_steps=100 \
        OUTPUTMODELDIR="$A1_OUTDIR" \
        postfix="a1fr" \
        "hydra.run.dir=outputs/tune_log/dual_f05_a1finerun/\${now:%Y-%m-%d_%H-%M-%S}"

    A1_RUNDIR=$(find "$A1_OUTDIR" -name "fullmodel" -type d | head -1 | xargs dirname)
fi

[ -z "$A1_RUNDIR" ] && { echo "ERROR: cannot locate A1 finerun dir"; exit 1; }
echo "A1 finerun dir: $A1_RUNDIR"

# ---------- Phase 2: eval requested fine-grained ckpts ----------
echo "============================================================"
echo "Phase 2: evaluating ckpts: $CKPT_STEPS"
echo "============================================================"

RESULTS_FILE="/tmp/forget05_fine_scan_results.csv"
echo "step,FQ,MU,forget_proba,forget_rouge,retain_rouge" > "$RESULTS_FILE"

run_eval() {
    local STEP=$1
    local A1_CKPT="$A1_RUNDIR/checkpoint-$STEP"
    local TAG="fine${STEP}"

    if [ ! -d "$A1_CKPT" ]; then
        echo "  [step=$STEP] missing ckpt — skip"
        echo "$STEP,,,,," >> "$RESULTS_FILE"
        return
    fi

    echo "  [$(date '+%H:%M:%S')] eval step=$STEP"

    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/eval_tofu.py \
        data=tofu \
        data.dataset.split=$SPLIT \
        data.dataset.eval.retain_result="data/forget05_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json" \
        data.dataset.eval.batch_size=4 \
        model=tofu-llama-2 \
        model_mode=dual_uld \
        model_mode.weight_a1=-0.8 \
        model_mode.weight_a2=0.8 \
        model_mode.top_logit_filter=1e-2 \
        model_mode.a1_ckpt_path="$A1_CKPT" \
        model_mode.a2_ckpt_path="$A2_CKPT" \
        ckpt_path="$A1_CKPT" \
        OUTDIRNAME="outputs_eval/f05_fine_scan/$TAG" \
        "hydra.run.dir=outputs/tune_log/fine_${TAG}/\${now:%Y-%m-%d_%H-%M-%S}" \
        > "/tmp/fine_${TAG}.log" 2>&1

    local CSV
    CSV=$(find "outputs/tune_log/fine_${TAG}" -name "aggregate_stat.csv" | head -1)
    if [ -n "$CSV" ]; then
        local FQ MU FP FR RR
        FQ=$(awk -F',' 'NR==2 {printf "%.4f", $14}' "$CSV")
        MU=$(awk -F',' 'NR==2 {printf "%.4f", $13}' "$CSV")
        FP=$(awk -F',' 'NR==2 {printf "%.4f", $11}' "$CSV")
        FR=$(awk -F',' 'NR==2 {printf "%.4f", $10}' "$CSV")
        RR=$(awk -F',' 'NR==2 {printf "%.4f", $1}' "$CSV")
        echo "$STEP,$FQ,$MU,$FP,$FR,$RR" >> "$RESULTS_FILE"
        printf "    → FQ=%s  MU=%s  forget_proba=%s\n" "$FQ" "$MU" "$FP"
    else
        echo "$STEP,,,,," >> "$RESULTS_FILE"
        echo "    → (no CSV)"
    fi
}

for STEP in $CKPT_STEPS; do
    run_eval "$STEP"
done

# ---------- Phase 3: summary ----------
echo
echo "============================================================"
echo "Done. Results in $RESULTS_FILE"
echo "Top 5 by FQ:"
echo "============================================================"
awk -F',' 'NR>1 && $2!="" {print $2"\t"$3"\t"$1}' "$RESULTS_FILE" \
    | sort -rn | head -5 \
    | awk -F'\t' '{printf "  step=%s  FQ=%s  MU=%s\n",$3,$1,$2}'

echo
echo "Full table:"
column -s, -t "$RESULTS_FILE"
