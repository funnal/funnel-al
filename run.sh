#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TF_CPP_MIN_LOG_LEVEL=2
export LIGHTEA_FAISS_GPU=1

PYTHON=${PYTHON:-python}

MODEL=${1:-duala}
DATASET=${2:-zh_en}
SEED=${3:-42}

echo "============================================================"
echo " FUNNEL-AL"
echo " Model:   $MODEL"
echo " Dataset: $DATASET"
echo " Seed:    $SEED"
echo " GPU:     $CUDA_VISIBLE_DEVICES"
echo "============================================================"

$PYTHON train.py \
    --data_path "data/${DATASET}/" \
    --model "$MODEL" \
    --seed "$SEED" \
    --simulate_oracle \
    --target_ratio 0.30 \
    --init_ratio 0.05 \
    --step_ratio 0.05 \
    --rounds 10 \
    --alpha 4.0 \
    --funnel_gamma 3.0 \
    --cov_eta 20.0 \
    --funnel_topk 10 \
    --inst_topk 10 \
    --inst_lambda 0.5 \
    --epochs_first 20 \
    --batch_size 1024 \
    --node_hidden 128 \
    --depth 2

echo ""
echo "Done. Results saved to results/${DATASET}/${MODEL}/seed${SEED}/"
