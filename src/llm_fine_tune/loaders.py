"""Shared output utilities for every pipeline phase.

Centralises the zstd Parquet write and the precondition-file guard so neither
is duplicated across the dataset, tokenizer, and publish scripts.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

OUTPUT_DIR = Path("output")


def write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — {hint}")
