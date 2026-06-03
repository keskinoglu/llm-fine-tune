"""Local caching layer for external data sources.

Provides two primitives: cloning/updating a git repository to a local directory,
and downloading an HuggingFace dataset to a local Parquet file. Both skip the
network on subsequent runs; pass the relevant refresh/update flag to force
a fresh fetch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl

DATA_DIR = Path("data")


def ensure_git_repo(url: str, local_dir: Path, *, update: bool = False) -> None:
    """Ensure url is cloned to local_dir; pull latest if update=True."""
    if local_dir.exists():
        if not update:
            print(f"Using existing clone at {local_dir}")
            return
        print(f"Pulling latest changes in {local_dir} ...")
        subprocess.run(["git", "-C", str(local_dir), "pull"], check=True)
        return
    print(f"Cloning {url} into {local_dir} ...")
    subprocess.run(["git", "clone", url, str(local_dir)], check=True)


def load_hf_dataset_cached(
    repo_id: str,
    cache_path: Path,
    *,
    refresh: bool = False,
) -> pl.DataFrame:
    """Return a Polars DataFrame from an HF dataset, using cache_path as a local Parquet cache.

    Downloads all splits and caches them on first use; reads from the local Parquet on
    subsequent calls. Pass refresh=True to force a fresh download regardless of cache state.
    The full dataset is cached so column selection can vary between calls without re-downloading.
    """
    if cache_path.exists() and not refresh:
        print(f"Using cached {repo_id} at {cache_path}")
        return pl.read_parquet(cache_path)

    print(f"Downloading {repo_id} ...")
    from datasets import concatenate_datasets, load_dataset

    dataset = load_dataset(repo_id)
    frame = concatenate_datasets(list(dataset.values())).to_polars()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cache_path, compression="zstd")
    print(f"Cached {frame.height:,} rows to {cache_path}")
    return frame
