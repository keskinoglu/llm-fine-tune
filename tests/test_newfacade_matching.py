from __future__ import annotations

import polars as pl
import pytest

from llm_fine_tune.dataset.newfacade_source import (
    slugify,
    title_mismatches,
    unmatched_problem_ids,
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
    nf_ids = {1, 2, 3}
    in_nf_only, in_base_only = unmatched_problem_ids(base_ids, nf_ids)
    assert in_nf_only == set()
    assert in_base_only == set()


def test_unmatched_nf_has_extra():
    base_ids = {1, 2}
    nf_ids = {1, 2, 3, 4}
    in_nf_only, in_base_only = unmatched_problem_ids(base_ids, nf_ids)
    assert in_nf_only == {3, 4}
    assert in_base_only == set()


def test_unmatched_base_has_extra():
    base_ids = {1, 2, 3, 99}
    nf_ids = {1, 2, 3}
    in_nf_only, in_base_only = unmatched_problem_ids(base_ids, nf_ids)
    assert in_nf_only == set()
    assert in_base_only == {99}


def test_unmatched_both_have_extra():
    base_ids = {1, 2, 10}
    nf_ids = {1, 2, 20}
    in_nf_only, in_base_only = unmatched_problem_ids(base_ids, nf_ids)
    assert in_nf_only == {20}
    assert in_base_only == {10}


# ---------------------------------------------------------------------------
# title_mismatches
# ---------------------------------------------------------------------------


def _base(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"problem_id": pl.Int64, "title": pl.Utf8})


def _nf(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"problem_id": pl.Int64, "task_id": pl.Utf8})


def test_title_mismatches_none():
    base = _base(
        [
            {"problem_id": 1, "title": "Two Sum"},
            {"problem_id": 2, "title": "Add Two Numbers"},
        ]
    )
    nf = _nf(
        [
            {"problem_id": 1, "task_id": "two-sum"},
            {"problem_id": 2, "task_id": "add-two-numbers"},
        ]
    )
    assert title_mismatches(base, nf) == []


def test_title_mismatches_detects_divergence():
    base = _base(
        [
            {"problem_id": 1, "title": "Two Sum"},
            {"problem_id": 2, "title": "Wrong Title"},
        ]
    )
    nf = _nf(
        [
            {"problem_id": 1, "task_id": "two-sum"},
            {"problem_id": 2, "task_id": "add-two-numbers"},
        ]
    )
    result = title_mismatches(base, nf)
    assert len(result) == 1
    assert result[0]["problem_id"] == 2
    assert result[0]["base_title"] == "Wrong Title"
    assert result[0]["task_id"] == "add-two-numbers"


def test_title_mismatches_unmatched_ids_ignored():
    """Ids present in only one source do not appear in mismatches (inner join)."""
    base = _base(
        [
            {"problem_id": 1, "title": "Two Sum"},
            {"problem_id": 99, "title": "Base Only"},
        ]
    )
    nf = _nf(
        [
            {"problem_id": 1, "task_id": "two-sum"},
            {"problem_id": 100, "task_id": "nf-only"},
        ]
    )
    assert title_mismatches(base, nf) == []
