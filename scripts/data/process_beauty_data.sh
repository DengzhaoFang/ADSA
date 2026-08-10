#!/bin/bash
# Process Beauty dataset with 5-core filtering
# Only generate embeddings for filtered items (default behavior)
# Add --generate_all_embeddings flag if you need ALL items embeddings for tokenizer training

cd ../../data

python process_amazon.py \
    --dataset Beauty \
    --review_path ../dataset/Amazon-Beauty/reviews_Beauty.json.gz \
    --meta_path ../dataset/Amazon-Beauty/meta_Beauty.json.gz \
    --output_dir ../dataset/Amazon-Beauty/processed/beauty-adsa-sentenceT5base \
    --min_interactions 5 \
    --embed_mode adsa \
    --embed_model sentence-t5 \
    --model_source modelscope \
    --device auto \
    --print_samples 10

echo ""
echo "=================================="
echo "Data processing completed!"

