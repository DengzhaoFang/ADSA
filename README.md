# ADSA

This repository contains the reference implementation for **ADSA**, a generative recommendation framework built on semantic IDs.

ADSA has two main stages:

1. **Stage 1: semantic ID tokenizer**
   - Trains an RQ-VAE tokenizer over item semantic and collaborative features.
   - Uses **PASA** (Popularity-Aware Soft Alignment) to replace hard in-batch contrastive targets with topology/text-aware soft targets.
   - Exports semantic ID mappings and purified continuous item features.
2. **Stage 2: generative recommender**
   - Trains an autoregressive sequential recommender over semantic IDs.
   - Uses **SSM** (Sparse Semantic Management) to inject purified continuous features through sparse MoE fusion during generation.

The repository also includes scripts for data preprocessing, LightGCN collaborative feature training, ADSA training, and several semantic-ID baselines.

## Repository Layout

```text
data/
  process_amazon.py              # Amazon review/meta preprocessing and text embedding generation
scripts/
  data/                          # Dataset preprocessing scripts
  lightGCN/                      # Stage-0 collaborative embedding scripts
  adsa/                          # ADSA Stage-1 tokenizer scripts
  TIGER/ EAGER/ LETTER/ ActionPiece/
                                 # Baseline tokenizer/recommender scripts
src/
  sid_tokenizer/
    adsa/                        # ADSA tokenizer, PASA loss/prior, semantic ID export
    lightgcn/                    # LightGCN used to initialize collaborative item features
  recommender/
    adsa/                        # ADSA generative recommender, SSM/MoE fusion, evaluation
```

## Environment

Python 3.10 is recommended. The project uses PyTorch 2.6.0 with CUDA 12.4 wheels in `pyproject.toml`.

Using `uv`:

```bash
uv sync
source .venv/bin/activate
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For CPU-only preprocessing, set `--device cpu` in the data scripts or call `data/process_amazon.py` directly.

## Data

The experiments use public Amazon review datasets. Place the raw gzipped review and metadata files under `dataset/` with the following structure:

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
```

Then run preprocessing:

```bash
bash scripts/data/process_beauty_data.sh
bash scripts/data/process_cds_data.sh
bash scripts/data/process_sports_data.sh
bash scripts/data/process_toys_data.sh
```

Each script applies 5-core filtering, creates `train.parquet`, `valid.parquet`, `test.parquet`, item ID mappings, and `item_emb.parquet`.

Default processed data directories are:

```text
dataset/Amazon-Beauty/processed/beauty-adsa-sentenceT5base/Beauty
dataset/Amazon-CDs/processed/cds-adsa-sentenceT5base/CDs
dataset/Amazon-Sports/processed/sports-adsa-sentenceT5base/Sports
dataset/Amazon-Toys/processed/toys-adsa-sentenceT5base/Toys
```

If you already have processed `*-tiger-sentenceT5base` directories from earlier experiments, pass `DATA_PATH=...` to the tokenizer scripts and `--sequence_data_path ...` to the recommender.

## Stage 0: LightGCN Collaborative Features

ADSA Stage 1 expects a LightGCN item embedding file at:

```text
<processed_dataset>/lightgcn/item_embeddings_collab.npy
```

Generate it with:

```bash
bash scripts/lightGCN/train_lightGCN_beauty.sh
bash scripts/lightGCN/train_lightGCN_cds.sh
bash scripts/lightGCN/train_lightGCN_sports.sh
bash scripts/lightGCN/train_lightGCN_toys.sh
```

The default LightGCN configuration uses embedding dimension 64, 3 GCN layers, 500 epochs, batch size 2048, learning rate 0.001, and early stopping on `Recall@20` with patience 20.

## Stage 1: ADSA Tokenizer with PASA

Run the dataset-specific scripts:

```bash
bash scripts/adsa/train_adsa_beauty.sh
bash scripts/adsa/train_adsa_cds.sh
bash scripts/adsa/train_adsa_sports.sh
bash scripts/adsa/train_adsa_toys.sh
```

Useful overrides:

```bash
DEVICE=cuda:1 bash scripts/adsa/train_adsa_beauty.sh
DATA_PATH=dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty bash scripts/adsa/train_adsa_beauty.sh
ALIGNMENT=cma bash scripts/adsa/train_adsa_beauty.sh
PASA_TOPK=7 TEXT_SHARPEN_GAMMA=5.0 GRAPH_SCALE_BETA=0.05 bash scripts/adsa/train_adsa_cds.sh
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

Tokenizer early stopping monitors training `total_loss`; lower is better. The scripts use `early_stop_min_delta=1e-5`, warmup 5 epochs, cooldown 3 epochs, and patience 50 for Beauty or 30 for CDs/Sports/Toys.

## Stage 2: ADSA Recommender with SSM

Example for Beauty:

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

For CDs with the paper MoE setting:

```bash
python -m src.recommender.adsa.train \
  --config cds \
  --device cuda:0 \
  --num_workers 4 \
  --model_type t5-tiny-2 \
  --use_multimodal_fusion \
  --fusion_gate_type moe \
  --moe_num_experts 5 \
  --moe_top_k 2 \
  --moe_use_load_balancing \
  --moe_load_balance_weight 0.01 \
  --use_trie_constraints \
  --output_keywords adsa_ssm
```

Recommender early stopping monitors validation `NDCG@20`; higher is better. The default recommender config evaluates every 3 epochs and uses patience 10 for the compact T5 variants.

## Main ADSA Settings

The table below lists the default settings encoded in the ADSA scripts/configs.

| Dataset | PASA K | PASA beta | PASA gamma | lambda_PASA | Stage-1 output | SSM experts/top-k |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Beauty | 5 | 0.10 | 3.0 | 0.1 | `scripts/output/adsa_tokenizer/beauty/hparam_stage1_PASCL/pasa_text_dominant` | 3/2 |
| CDs | 7 | 0.05 | 5.0 | 0.1 | `scripts/output/adsa_tokenizer/cds/hparam_stage1_PASCL/pasa_graph_dominant` | 5/2 |
| Sports | 5 | 0.10 | 3.0 | 0.1 | `scripts/output/adsa_tokenizer/sports/hparam_stage1_PASCL/pasa_text_dominant` | 3/2 |
| Toys | 5 | 0.05 | 3.0 | 0.1 | `scripts/output/adsa_tokenizer/toys/hparam_stage1_PASCL/pasa_default` | 3/2 |

Here `K` is PASA top-k truncation, `beta` is the graph similarity scaling threshold, and `gamma` is the text-similarity sharpening exponent.

## Reproducibility Notes

- Random seed defaults to 42 in the recommender config and k-means initialization.
- Stage-1 target construction is offline preprocessing inside tokenizer training; PASA targets are not used in the online serving path.
- Text embeddings are cached in `item_emb.parquet`; graph affinities are computed from `train.parquet` and LightGCN embeddings are stored under the processed dataset directory.
- Logs and checkpoints are written under `scripts/output/`.
- To resume recommender training, use `--resume <checkpoint.pt>`. To resume Stage-1 tokenizer training, use `--resume <checkpoint.pt>` with `src/sid_tokenizer/adsa/train_adsa.py`.

## Baselines

Baseline entry points are provided under:

```text
scripts/TIGER/
scripts/EAGER/
scripts/LETTER/
scripts/ActionPiece/
```

Each baseline has tokenizer and/or recommender scripts mirroring the ADSA data layout.

## Troubleshooting

- If `item_emb.parquet` contains `attribute_embedding` rather than `embedding`, ADSA Stage 1 now reads it automatically.
- If Stage 1 reports a missing collaborative embedding file, run the matching LightGCN script first.
- If you use preprocessed data from another directory, override both the tokenizer `DATA_PATH` and the recommender `--sequence_data_path`.
- If ModelScope or HuggingFace download is unavailable, manually place `sentence-t5-base` at `./sentence-t5-base`.
