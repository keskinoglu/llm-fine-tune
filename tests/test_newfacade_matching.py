from __future__ import annotations

import polars as pl
import pytest

from llm_fine_tune.dataset.source_newfacade import (
    slugify,
    title_mismatches,
    unmatched_parallel_ids,
)


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Two Sum", "two-sum"),
        ("Add Two Numbers", "add-two-numbers"),
        ("3Sum", "3sum"),
        ("4Sum II", "4sum-ii"),
        ("N-Queens", "n-queens"),
        ("N-Queens II", "n-queens-ii"),
        ("Longest Common Prefix", "longest-common-prefix"),
        ("Two Sum II - Input Array Is Sorted", "two-sum-ii-input-array-is-sorted"),
        ("Valid Parentheses", "valid-parentheses"),
    ],
)
def test_slugify(title: str, expected: str):
    assert slugify(title) == expected


# ---------------------------------------------------------------------------
# unmatched_problem_ids
# ---------------------------------------------------------------------------


def test_unmatched_all_matched():
    base_ids = {1, 2, 3}
    newfacade_ids = {1, 2, 3}
    only_in_newfacade, only_in_base = unmatched_parallel_ids(base_ids, newfacade_ids)
    assert only_in_newfacade == set()
    assert only_in_base == set()


def test_unmatched_newfacade_has_extra():
    base_ids = {1, 2}
    newfacade_ids = {1, 2, 3, 4}
    only_in_newfacade, only_in_base = unmatched_parallel_ids(base_ids, newfacade_ids)
    assert only_in_newfacade == {3, 4}
    assert only_in_base == set()


def test_unmatched_base_has_extra():
    base_ids = {1, 2, 3, 99}
    newfacade_ids = {1, 2, 3}
    only_in_newfacade, only_in_base = unmatched_parallel_ids(base_ids, newfacade_ids)
    assert only_in_newfacade == set()
    assert only_in_base == {99}


def test_unmatched_both_have_extra():
    base_ids = {1, 2, 10}
    newfacade_ids = {1, 2, 20}
    only_in_newfacade, only_in_base = unmatched_parallel_ids(base_ids, newfacade_ids)
    assert only_in_newfacade == {20}
    assert only_in_base == {10}


# ---------------------------------------------------------------------------
# title_mismatches
# ---------------------------------------------------------------------------


def _base_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"parallel_id": pl.Int64, "title": pl.Utf8})


def _newfacade_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"parallel_id": pl.Int64, "task_id": pl.Utf8})


def test_title_mismatches_none():
    base = _base_frame(
        [
            {"parallel_id": 1, "title": "Two Sum"},
            {"parallel_id": 2, "title": "Add Two Numbers"},
        ]
    )
    newfacade = _newfacade_frame(
        [
            {"parallel_id": 1, "task_id": "two-sum"},
            {"parallel_id": 2, "task_id": "add-two-numbers"},
        ]
    )
    assert title_mismatches(base, newfacade) == []


def test_title_mismatches_detects_divergence():
    base = _base_frame(
        [
            {"parallel_id": 1, "title": "Two Sum"},
            {"parallel_id": 2, "title": "Wrong Title"},
        ]
    )
    newfacade = _newfacade_frame(
        [
            {"parallel_id": 1, "task_id": "two-sum"},
            {"parallel_id": 2, "task_id": "add-two-numbers"},
        ]
    )
    result = title_mismatches(base, newfacade)
    assert len(result) == 1
    assert result[0]["parallel_id"] == 2
    assert result[0]["base_title"] == "Wrong Title"
    assert result[0]["task_id"] == "add-two-numbers"


def test_title_mismatches_unmatched_ids_ignored():
    """Ids present in only one source do not appear in mismatches (inner join)."""
    base = _base_frame(
        [
            {"parallel_id": 1, "title": "Two Sum"},
            {"parallel_id": 99, "title": "Base Only"},
        ]
    )
    newfacade = _newfacade_frame(
        [
            {"parallel_id": 1, "task_id": "two-sum"},
            {"parallel_id": 100, "task_id": "newfacade-only"},
        ]
    )
    assert title_mismatches(base, newfacade) == []
