# -*- coding: utf-8 -*-

import argparse
import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from process_amazon import (
    EMBEDDING_MODELS,
    apply_kcore_filter,
    download_model_if_needed,
    select_device,
)


def setup_logging(output_dir, dataset_name):
    log_file = os.path.join(output_dir, f"{dataset_name}_preprocessing.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("movielens_preprocessing")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Process MovieLens dataset for ADSA")
    parser.add_argument("--dataset", type=str, default="ML-1M")
    parser.add_argument("--ratings_path", type=str, required=True)
    parser.add_argument("--movies_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument(
        "--embed_model",
        type=str,
        default="sentence-t5",
        choices=list(EMBEDDING_MODELS.keys()),
    )
    parser.add_argument(
        "--model_source",
        type=str,
        default="modelscope",
        choices=["huggingface", "modelscope"],
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--print_samples", type=int, default=10)
    return parser.parse_args()


def load_ratings(path):
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            rows.append(
                {
                    "userID": user_id,
                    "itemID": movie_id,
                    "rating": float(rating),
                    "timestamp": int(timestamp),
                }
            )
    return pd.DataFrame(rows)


def load_movies(path):
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.rstrip("\n").split("::")
            rows.append(
                {
                    "raw_item_id": movie_id,
                    "title": title,
                    "genres": genres.replace("|", ", "),
                }
            )
    return pd.DataFrame(rows)


def split_sequences(filtered_df, user_mapping, item_mapping, output_path, logger):
    user_sequences = {}
    for _, row in filtered_df.iterrows():
        user_id = user_mapping[row["userID"]]
        item_id = item_mapping[row["itemID"]]
        user_sequences.setdefault(user_id, []).append((item_id, row["timestamp"]))

    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[1])
        user_sequences[user_id] = [item for item, _ in user_sequences[user_id]]

    train_data, val_data, test_data = [], [], []
    for user_id, item_sequence in user_sequences.items():
        seq_len = len(item_sequence)
        if seq_len < 3:
            continue
        if seq_len > 3:
            history = item_sequence[:-3]
            if history:
                train_data.append(
                    {"user": user_id, "history": history, "target": item_sequence[-3]}
                )
        val_data.append(
            {"user": user_id, "history": item_sequence[:-2], "target": item_sequence[-2]}
        )
        test_data.append(
            {"user": user_id, "history": item_sequence[:-1], "target": item_sequence[-1]}
        )

    pd.DataFrame(train_data).to_parquet(os.path.join(output_path, "train.parquet"), index=False)
    pd.DataFrame(val_data).to_parquet(os.path.join(output_path, "valid.parquet"), index=False)
    pd.DataFrame(test_data).to_parquet(os.path.join(output_path, "test.parquet"), index=False)

    logger.info("Data split completed:")
    logger.info(f"  Train: {len(train_data):,}")
    logger.info(f"  Valid: {len(val_data):,}")
    logger.info(f"  Test:  {len(test_data):,}")


def build_item_embeddings(movies_df, item_mapping, item_counts, args, logger):
    device = select_device(args.device, logger)
    model_path = download_model_if_needed(args.embed_model, args.model_source, logger)
    model = SentenceTransformer(model_path, device=device)

    reverse_mapping = {raw_id: mapped_id for raw_id, mapped_id in item_mapping.items()}
    movie_lookup = movies_df.set_index("raw_item_id").to_dict(orient="index")

    rows = []
    for raw_id, item_id in sorted(reverse_mapping.items(), key=lambda x: x[1]):
        info = movie_lookup.get(str(raw_id), {})
        title = info.get("title", "")
        genres = info.get("genres", "")
        text = f"title: {title}\ngenres: {genres}"
        embedding = model.encode(text)
        rows.append(
            {
                "ItemID": item_id,
                "title": title,
                "genres": genres,
                "embedding": embedding.tolist(),
                "interaction_count": item_counts.get(item_id, 0),
            }
        )

    item_df = pd.DataFrame(rows)
    item_df["popularity_log"] = np.log1p(item_df["interaction_count"])
    max_log = item_df["popularity_log"].max()
    min_log = item_df["popularity_log"].min()
    if max_log > min_log:
        item_df["popularity_score"] = (
            item_df["popularity_log"] - min_log
        ) / (max_log - min_log)
    else:
        item_df["popularity_score"] = 0.5

    logger.info(f"Generated item embeddings: {item_df.shape}")
    for _, row in item_df.head(args.print_samples).iterrows():
        logger.info(f"  Item {row['ItemID']}: {row['title']} [{row['genres']}]")
    return item_df


def main():
    args = parse_args()
    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logging(output_path, args.dataset)

    logger.info("=" * 80)
    logger.info("MOVIELENS DATASET PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    ratings_df = load_ratings(args.ratings_path)
    movies_df = load_movies(args.movies_path)
    logger.info(f"Loaded ratings: {len(ratings_df):,}")
    logger.info(f"Loaded movies: {len(movies_df):,}")

    filtered_df, stats = apply_kcore_filter(ratings_df, args.min_interactions)
    logger.info(f"Filtering iterations: {stats['iterations']}")
    logger.info(f"Filtered interactions: {len(filtered_df):,}")

    unique_users = sorted(filtered_df["userID"].unique())
    unique_items = sorted(filtered_df["itemID"].unique())
    user_mapping = {user: idx + 1 for idx, user in enumerate(unique_users)}
    item_mapping = {item: idx + 1 for idx, item in enumerate(unique_items)}
    np.save(os.path.join(output_path, "user_mapping.npy"), user_mapping)
    np.save(os.path.join(output_path, "item_mapping.npy"), item_mapping)

    split_sequences(filtered_df, user_mapping, item_mapping, output_path, logger)

    item_counts = Counter()
    for _, row in filtered_df.iterrows():
        item_counts[item_mapping[row["itemID"]]] += 1

    item_df = build_item_embeddings(movies_df, item_mapping, item_counts, args, logger)
    item_df.to_parquet(os.path.join(output_path, "item_emb.parquet"), index=False)

    logger.info("Preprocessing completed.")
    logger.info(f"Output directory: {output_path}")


if __name__ == "__main__":
    main()
