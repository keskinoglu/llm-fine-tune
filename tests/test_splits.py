from __future__ import annotations

import polars as pl
import pytest

from llm_fine_tune.dataset.splits import split_by_key


def _make_df(n_snippets: int, rows_per_snippet: int = 6) -> pl.DataFrame:
    rows = [
        {"parallel_id": cid, "value": f"row-{cid}-{i}"}
        for cid in range(1, n_snippets + 1)
        for i in range(rows_per_snippet)
    ]
    return pl.DataFrame(rows)


def test_split_is_deterministic():
    df = _make_df(100)
    train1, test1 = split_by_key(df, "parallel_id", 0.30, seed=0)
    train2, test2 = split_by_key(df, "parallel_id", 0.30, seed=0)
    assert train1.equals(train2)
    assert test1.equals(test2)


def test_split_different_seeds_differ():
    df = _make_df(100)
    train1, _ = split_by_key(df, "parallel_id", 0.30, seed=0)
    train2, _ = split_by_key(df, "parallel_id", 0.30, seed=42)
    assert not train1.equals(train2)


def test_split_keys_are_disjoint():
    df = _make_df(100)
    train, test = split_by_key(df, "parallel_id", 0.30, seed=0)
    train_ids = set(train["parallel_id"].to_list())
    test_ids = set(test["parallel_id"].to_list())
    assert train_ids.isdisjoint(test_ids)


def test_split_all_rows_accounted_for():
    df = _make_df(100)
    train, test = split_by_key(df, "parallel_id", 0.30, seed=0)
    assert train.height + test.height == df.height


def test_split_ratio_is_approximate():
    df = _make_df(200)
    _, test = split_by_key(df, "parallel_id", 0.30, seed=0)
    unique_test = test["parallel_id"].n_unique()
    ratio = unique_test / 200
    assert 0.25 <= ratio <= 0.35, f"test fraction {ratio:.2f} outside expected range"


@pytest.mark.parametrize("test_frac", [0.1, 0.3, 0.5])
def test_split_both_sides_non_empty(test_frac: float):
    df = _make_df(20)
    train, test = split_by_key(df, "parallel_id", test_frac, seed=0)
    assert train.height > 0
    assert test.height > 0
