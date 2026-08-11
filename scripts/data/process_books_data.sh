#!/bin/bash
# Process Amazon Books dataset with 5-core filtering.

cd ../../data

python process_amazon.py \
    --dataset Books \
    --review_path ../dataset/Amazon-Books/reviews_Books.json.gz \
    --meta_path ../dataset/Amazon-Books/meta_Books.json.gz \
    --output_dir ../dataset/Amazon-Books/processed/books-adsa-sentenceT5base \
    --min_interactions 5 \
    --embed_mode adsa \
    --embed_model sentence-t5 \
    --model_source modelscope \
    --device auto \
    --print_samples 10

echo ""
echo "=================================="
echo "Data processing completed!"
