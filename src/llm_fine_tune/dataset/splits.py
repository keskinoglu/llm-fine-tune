from __future__ import annotations

import random

import polars as pl


def split_by_key(
    df: pl.DataFrame,
    key_col: str,
    test_frac: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split df into (train, test) by unique values of key_col.

    All rows sharing a key value land on the same side, preventing cross-key leakage.
    Identical (seed, test_frac, key values) always produce identical partitions.
    """
    keys = df[key_col].unique().sort().to_list()
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_test = max(1, round(len(keys) * test_frac))
    test_keys = set(keys[:n_test])
    train_keys = set(keys[n_test:])
    return (
        df.filter(pl.col(key_col).is_in(list(train_keys))),
        df.filter(pl.col(key_col).is_in(list(test_keys))),
    )
