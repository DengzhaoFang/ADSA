#!/bin/bash
# Process MovieLens-1M.

cd ../../data

python process_movielens.py \
    --dataset ML-1M \
    --ratings_path ../dataset/MovieLens-1M/ml-1m/ratings.dat \
    --movies_path ../dataset/MovieLens-1M/ml-1m/movies.dat \
    --output_dir ../dataset/MovieLens-1M/processed/ml1m-adsa-sentenceT5base \
    --min_interactions 5 \
    --embed_model sentence-t5 \
    --model_source modelscope \
    --device auto \
    --print_samples 10

echo ""
echo "=================================="
echo "Data processing completed!"
