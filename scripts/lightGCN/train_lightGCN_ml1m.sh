#!/bin/bash

cd ../../src/sid_tokenizer/lightgcn

DATA_DIR="../../../dataset/MovieLens-1M/processed/ml1m-adsa-sentenceT5base/ML-1M"
OUTPUT_DIR=$DATA_DIR/lightgcn
EXP_NAME="${EXP_NAME:-}"

EMBEDDING_DIM=64
N_LAYERS=3
N_EPOCHS=500
BATCH_SIZE=2048
LR=0.001
REG_WEIGHT=0.0001
EARLY_STOP_PATIENCE=20
EVAL_EVERY=2
EVAL_BATCH_SIZE=1024
EARLY_STOP_METRIC="Recall@20"
K_VALUES="5 10 20"
SAVE_EVERY=10
DEVICE="cuda"
GPU_ID=2
USE_VAL=""

python train.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --embedding_dim $EMBEDDING_DIM \
    --n_layers $N_LAYERS \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --reg_weight $REG_WEIGHT \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --eval_every $EVAL_EVERY \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --early_stop_metric $EARLY_STOP_METRIC \
    --k_values $K_VALUES \
    --save_every $SAVE_EVERY \
    --device $DEVICE \
    ${GPU_ID:+--gpu_id $GPU_ID} \
    $USE_VAL
