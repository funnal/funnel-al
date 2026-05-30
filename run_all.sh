#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TF_CPP_MIN_LOG_LEVEL=2
export LIGHTEA_FAISS_GPU=1

PYTHON=${PYTHON:-python}

SEEDS="42 123 456"
DATASETS="zh_en ja_en fr_en en_fr_15k_V1 en_de_15k_V1"
MODELS="duala lightea gcn_align"

for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        for SEED in $SEEDS; do
            echo ">>> $MODEL | $DATASET | seed=$SEED"
            $PYTHON train.py \
                --data_path "data/${DATASET}/" \
                --model "$MODEL" \
                --seed "$SEED" \
                --simulate_oracle \
                --quiet
        done
    done
done

echo "All experiments complete."
