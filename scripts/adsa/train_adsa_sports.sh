#!/usr/bin/env bash
set -euo pipefail

# ADSA Stage 1 tokenizer training for Amazon Sports.
# Run from anywhere:
#   bash scripts/adsa/train_adsa_sports.sh

cd "$(dirname "$0")/../.."
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

DATA_PATH="${DATA_PATH:-dataset/Amazon-Sports/processed/sports-adsa-sentenceT5base/Sports}"
OUTPUT_DIR="${OUTPUT_DIR:-scripts/output/adsa_tokenizer/sports/hparam_stage1_PASCL/pasa_text_dominant}"
DEVICE="${DEVICE:-cuda}"
ALIGNMENT="${ALIGNMENT:-pasa}"

MODEL_ARGS=(
    --n_layers 3
    --n_embed_per_layer "256,256,256"
    --latent_dim 32
    --content_dim 768
    --collab_dim 64
    --ide on
    --ide_dim 128
)

TRAIN_ARGS=(
    --epochs 500
    --batch_size 512
    --learning_rate 1e-4
    --weight_decay 1e-4
    --grad_clip 1.0
    --commit_weight 0.25
    --use_ema
    --ema_decay 0.99
    --quantize_mode rotation
    --use_scheduler
    --scheduler_type warmup_cosine
    --warmup_ratio 0.1
    --early_stop_patience 30
    --early_stop_min_delta 1e-5
    --early_stop_cooldown 3
    --early_stop_warmup_epochs 5
    --perplexity_collapse_ratio 0.35
    --perplexity_collapse_patience 3
    --kmeans_init_samples 8192
    --save_every 50
    --num_workers 4
    --log_level INFO
    --device "$DEVICE"
)

case "$ALIGNMENT" in
    pasa)
        ALIGN_ARGS=(
            --use_pasa
            --lambda_pasa "${LAMBDA_PASA:-0.1}"
            --pasa_temperature "${PASA_TEMPERATURE:-0.2}"
            --pasa_topk "${PASA_TOPK:-5}"
            --text_sharpen_gamma "${TEXT_SHARPEN_GAMMA:-3.0}"
            --graph_scale_beta "${GRAPH_SCALE_BETA:-0.10}"
        )
        ;;
    cma)
        ALIGN_ARGS=(
            --lambda_cma "${LAMBDA_CMA:-0.1}"
            --cma_temperature "${CMA_TEMPERATURE:-0.07}"
        )
        ;;
    *)
        echo "Unknown ALIGNMENT=$ALIGNMENT. Use 'pasa' or 'cma'." >&2
        exit 2
        ;;
esac

DUAL_HEAD_ARGS=()
if [ "${USE_DUAL_HEAD:-0}" = "1" ]; then
    DUAL_HEAD_ARGS=(--use_dual_head --dual_head_pop_weight "${DUAL_HEAD_POP_WEIGHT:-true}")
fi

python src/sid_tokenizer/adsa/train_adsa.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    "${MODEL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${ALIGN_ARGS[@]}" \
    "${DUAL_HEAD_ARGS[@]}"
