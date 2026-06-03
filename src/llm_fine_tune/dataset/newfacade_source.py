from __future__ import annotations

import re

import polars as pl

from llm_fine_tune.dataset import sources

# Columns merged from newfacade into base.
# Excluded (Python-specific, unusable for C++/Java evaluation):
#   test         — Python assertion harness, language-specific
#   starter_code — Python stub only
#   completion   — Python reference solution (walkccc's python col already covers this)
MERGE_COLUMNS = [
    "difficulty",
    "input_output",
    "problem_description",
    "entry_point",
    "prompt",
    "query",
    "response",
    "tags",
    "estimated_date",
    "task_id",
]

_REPO_ID = "newfacade/LeetCodeDataset"
_CACHE_PATH = sources.DATA_DIR / "newfacade" / "leetcode-dataset.parquet"


def load_newfacade_frame(*, refresh: bool = False) -> pl.DataFrame:
    """Return a Polars frame keyed by problem_id carrying MERGE_COLUMNS.

    Uses a local parquet cache; pass refresh=True to force a fresh download.
    The full dataset is cached so adding columns later requires no re-download.
    """
    frame = sources.load_hf_dataset_cached(_REPO_ID, _CACHE_PATH, refresh=refresh)
    return frame.select(["question_id"] + MERGE_COLUMNS).rename(
        {"question_id": "problem_id"}
    )


# ---------------------------------------------------------------------------
# Pure reconciliation helpers (no I/O — fully unit-testable)
# ---------------------------------------------------------------------------


def slugify(title: str) -> str:
    """Normalise a LeetCode problem title to its URL slug form.

    Example: 'Two Sum' -> 'two-sum', 'N-Queens II' -> 'n-queens-ii'.
    """
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def unmatched_problem_ids(
    base_ids: set[int],
    nf_ids: set[int],
) -> tuple[set[int], set[int]]:
    """Return (in_nf_only, in_base_only): ids present in one source but not the other."""
    return nf_ids - base_ids, base_ids - nf_ids


def title_mismatches(
    base_df: pl.DataFrame,
    nf_df: pl.DataFrame,
) -> list[dict]:
    """Return matched rows where slugify(base.title) != newfacade.task_id."""
    joined = base_df.select(["problem_id", "title"]).join(
        nf_df.select(["problem_id", "task_id"]),
        on="problem_id",
        how="inner",
    )
    return [
        {
            "problem_id": row["problem_id"],
            "base_title": row["title"],
            "task_id": row["task_id"],
        }
        for row in joined.iter_rows(named=True)
        if slugify(row["title"]) != row["task_id"]
    ]
