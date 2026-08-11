# ADSA

This repository contains the reference implementation for **ADSA**, a generative recommendation framework based on semantic IDs.

ADSA contains two stages:

1. **Semantic ID tokenizer**: trains an RQ-VAE tokenizer with PASA (Popularity-Aware Soft Alignment) and exports semantic ID mappings plus purified item features.
2. **Generative recommender**: trains an autoregressive recommender over semantic IDs and uses SSM (Sparse Semantic Management) to inject continuous item features during generation.

## Repository Layout

```text
data/
  process_amazon.py              # Amazon preprocessing
  process_movielens.py           # MovieLens preprocessing
scripts/
  data/                          # Dataset preprocessing scripts
  lightGCN/                      # Collaborative embedding scripts
  adsa/                          # ADSA tokenizer scripts
  TIGER/ EAGER/ LETTER/ ActionPiece/
                                 # Baseline scripts
src/
  sid_tokenizer/adsa/            # ADSA tokenizer and PASA modules
  sid_tokenizer/lightgcn/        # LightGCN feature extraction
  recommender/adsa/              # ADSA recommender and SSM/MoE modules
```

## Environment

Python 3.10 is recommended.

```bash
uv sync
source .venv/bin/activate
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Data Preparation

The code supports Amazon review datasets and MovieLens-1M. Raw files are expected under `dataset/`.

```text
dataset/
  Amazon-Beauty/
    reviews_Beauty.json.gz
    meta_Beauty.json.gz
  Amazon-CDs/
    reviews_CDs_and_Vinyl.json.gz
    meta_CDs_and_Vinyl.json.gz
  Amazon-Sports/
    reviews_Sports_and_Outdoors.json.gz
    meta_Sports_and_Outdoors.json.gz
  Amazon-Toys/
    reviews_Toys_and_Games.json.gz
    meta_Toys_and_Games.json.gz
  Amazon-Books/
    reviews_Books.json.gz
    meta_Books.json.gz
  MovieLens-1M/
    ml-1m/
      ratings.dat
      movies.dat
```

Run preprocessing with:

```bash
bash scripts/data/process_beauty_data.sh
bash scripts/data/process_cds_data.sh
bash scripts/data/process_sports_data.sh
bash scripts/data/process_toys_data.sh
bash scripts/data/process_books_data.sh
bash scripts/data/process_ml1m_data.sh
```

The processed directories follow the same layout:

```text
dataset/Amazon-Beauty/processed/beauty-adsa-sentenceT5base/Beauty
dataset/Amazon-CDs/processed/cds-adsa-sentenceT5base/CDs
dataset/Amazon-Sports/processed/sports-adsa-sentenceT5base/Sports
dataset/Amazon-Toys/processed/toys-adsa-sentenceT5base/Toys
dataset/Amazon-Books/processed/books-adsa-sentenceT5base/Books
dataset/MovieLens-1M/processed/ml1m-adsa-sentenceT5base/ML-1M
```

Each processed dataset contains sequence splits and item-side embeddings used by the tokenizer and recommender.

## Collaborative Features

Train LightGCN item embeddings before Stage 1:

```bash
bash scripts/lightGCN/train_lightGCN_beauty.sh
bash scripts/lightGCN/train_lightGCN_cds.sh
bash scripts/lightGCN/train_lightGCN_sports.sh
bash scripts/lightGCN/train_lightGCN_toys.sh
bash scripts/lightGCN/train_lightGCN_books.sh
bash scripts/lightGCN/train_lightGCN_ml1m.sh
```

The resulting item features are stored under each processed dataset directory.

## ADSA Tokenizer

Train semantic IDs with:

```bash
bash scripts/adsa/train_adsa_beauty.sh
bash scripts/adsa/train_adsa_cds.sh
bash scripts/adsa/train_adsa_sports.sh
bash scripts/adsa/train_adsa_toys.sh
bash scripts/adsa/train_adsa_books.sh
bash scripts/adsa/train_adsa_ml1m.sh
```

Common overrides:

```bash
DEVICE=cuda:1 bash scripts/adsa/train_adsa_beauty.sh
ALIGNMENT=cma bash scripts/adsa/train_adsa_beauty.sh
DATA_PATH=/path/to/processed/Dataset OUTPUT_DIR=/path/to/output bash scripts/adsa/train_adsa_books.sh
```

Stage-1 outputs include:

```text
semantic_id_mappings.json
semantic_ids.npy
item_purified_content.npy
item_purified_collab.npy
item_purified_z_clean.npy
item_purified_ids.npy
item_codebook_zq.npy
training.log
```

## ADSA Recommender

Example:

```bash
python -m src.recommender.adsa.train \
  --config beauty \
  --device cuda:0 \
  --num_workers 4 \
  --model_type t5-tiny-2 \
  --use_multimodal_fusion \
  --fusion_gate_type moe \
  --moe_num_experts 3 \
  --moe_top_k 2 \
  --moe_use_load_balancing \
  --moe_load_balance_weight 0.01 \
  --use_trie_constraints \
  --output_keywords adsa_ssm
```

Supported `--config` values:

```text
beauty, cds, sports, toys, books, ml1m
```

For custom Stage-1 outputs:

```bash
python -m src.recommender.adsa.train \
  --config books \
  --semantic_mapping_path /path/to/semantic_id_mappings.json \
  --purified_content_path /path/to/item_purified_content.npy \
  --purified_collab_path /path/to/item_purified_collab.npy \
  --sequence_data_path /path/to/processed/Books \
  --use_multimodal_fusion
```
