"""
data/download_dataset.py
------------------------
Downloads the first 5 000 rows of the McAuley-Lab/Amazon-Reviews-2023
dataset (category: All_Beauty, split: train) and writes them to
data/reviews.csv.

Why this dataset?
    It is public, well-structured, and representative of the kind of
    e-commerce reviews our multi-agent pipeline will ultimately process.
    The All_Beauty subset is small enough to iterate on quickly while
    still offering varied sentiment.

Usage:
    python -m data.download_dataset
    # or
    python data/download_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Target location for the CSV — sibling to this script.
OUTPUT_CSV = Path(__file__).resolve().parent / "reviews.csv"

# HuggingFace dataset coordinates.
DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
CONFIG_NAME = "raw_review_All_Beauty"   # "raw_review_<category>" naming scheme
SPLIT = "full"                          # this dataset ships as a single split
N_ROWS = 5_000


def download() -> Path:
    """Stream the first N_ROWS of the dataset and persist them as CSV.

    We use streaming=True so we never materialise the full dataset in
    memory — we only need the first slice, and the full corpus is many
    GB. ``trust_remote_code`` is required by HuggingFace for this repo's
    loader script.
    """
    print(f"Downloading {DATASET_ID} / {CONFIG_NAME} (first {N_ROWS} rows)…")

    # Streaming dataset -> an IterableDataset we can ``take`` from.
    stream = load_dataset(
        DATASET_ID,
        CONFIG_NAME,
        split=SPLIT,
        streaming=True,
        trust_remote_code=True,
    )

    # ``take`` returns an IterableDataset limited to the first N rows.
    # Converting to a list materialises only those rows.
    rows = list(stream.take(N_ROWS))

    # pandas handles CSV escaping and column ordering for us.
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print(f"Columns: {list(df.columns)}")
    return OUTPUT_CSV


if __name__ == "__main__":
    download()
