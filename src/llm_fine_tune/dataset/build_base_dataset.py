"""Build the base LeetCode solutions Parquet dataset (Stage 1, step 1).

Reads problem solutions from the source repository, optionally enriches them
with problem metadata from a secondary dataset, then writes
output/leetcode-solutions.parquet.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import polars as pl

from llm_fine_tune import loaders
from llm_fine_tune.dataset import source_newfacade, source_walkccc

OUTPUT_PATH = loaders.OUTPUT_DIR / "leetcode-solutions.parquet"


def main() -> None:
    args = _parse_args()

    base_frame = source_walkccc.load_walkccc_frame(pull=args.pull)

    if args.enrich:
        newfacade_frame = source_newfacade.load_newfacade_frame(refresh=args.refresh)
        _print_integrity_report(base_frame, newfacade_frame)
        base_frame = _enrich(base_frame, newfacade_frame)

    loaders.write_parquet(base_frame, OUTPUT_PATH)
    _print_summary(base_frame)


# ---- Argument parsing ----


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the leetcode-solutions Parquet dataset."
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull latest changes from the walkccc/LeetCode source repo before building.",
    )
    parser.add_argument(
        "--enrich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enrich with problem metadata from the secondary dataset (default: on; use --no-enrich to skip).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download all cached HF datasets before building.",
    )
    return parser.parse_args()


# ---- Enrichment ----


def _enrich(base_frame: pl.DataFrame, secondary_frame: pl.DataFrame) -> pl.DataFrame:
    return base_frame.join(secondary_frame, on="parallel_id", how="left")


# ---- Integrity reporting ----


@dataclass(frozen=True)
class _SourceOverlapStats:
    primary_count: int
    secondary_count: int
    matched_count: int
    only_in_secondary: frozenset[int]
    only_in_primary: frozenset[int]


def _print_integrity_report(
    base_frame: pl.DataFrame,
    secondary_frame: pl.DataFrame,
) -> None:
    overlap = _calculate_source_overlap(base_frame, secondary_frame)
    _print_source_overlap(overlap)
    mismatches = source_newfacade.title_mismatches(base_frame, secondary_frame)
    _print_title_mismatches(mismatches)


def _calculate_source_overlap(
    base_frame: pl.DataFrame,
    secondary_frame: pl.DataFrame,
) -> _SourceOverlapStats:
    primary_ids = set(base_frame["parallel_id"].to_list())
    secondary_ids = set(secondary_frame["parallel_id"].to_list())
    only_in_secondary, only_in_primary = source_newfacade.unmatched_parallel_ids(
        primary_ids, secondary_ids
    )
    return _SourceOverlapStats(
        primary_count=len(primary_ids),
        secondary_count=len(secondary_ids),
        matched_count=len(primary_ids & secondary_ids),
        only_in_secondary=frozenset(only_in_secondary),
        only_in_primary=frozenset(only_in_primary),
    )


def _print_source_overlap(overlap: _SourceOverlapStats) -> None:
    print(
        f"\nIntegrity check: {overlap.primary_count:,} base code snippets, "
        f"{overlap.secondary_count:,} secondary code snippets"
    )
    print(f"  Matched:                {overlap.matched_count:,}")

    secondary_only_preview = sorted(overlap.only_in_secondary)[:10]
    ellipsis = "..." if len(overlap.only_in_secondary) > 10 else ""
    print(
        f"  In secondary only:      {len(overlap.only_in_secondary):,}"
        + (
            f" — ids: {secondary_only_preview}{ellipsis}"
            if overlap.only_in_secondary
            else ""
        )
    )
    print(f"  In base only:           {len(overlap.only_in_primary):,}")


def _print_title_mismatches(mismatches: list[dict]) -> None:
    if not mismatches:
        print("  Title mismatches:       0")
        return
    print(f"  Title mismatches:       {len(mismatches):,}")
    for mismatch in mismatches[:5]:
        print(
            f"    id={mismatch['parallel_id']}: "
            f"'{mismatch['base_title']}' vs task_id='{mismatch['task_id']}'"
        )
    if len(mismatches) > 5:
        print(f"    ... and {len(mismatches) - 5} more")


# ---- Summary ----


def _print_summary(base_frame: pl.DataFrame) -> None:
    print(f"\nSaved {base_frame.height:,} code snippets to {OUTPUT_PATH}")
    solution_counts = _count_solutions_per_language(base_frame)
    _print_solution_counts(solution_counts)
    if "difficulty" in base_frame.columns:
        difficulty_distribution = _calculate_difficulty_distribution(base_frame)
        _print_difficulty_distribution(difficulty_distribution)
    if "input_output" in base_frame.columns:
        covered, total = _calculate_input_output_coverage(base_frame)
        _print_input_output_coverage(covered, total)


def _count_solutions_per_language(base_frame: pl.DataFrame) -> dict[str, int]:
    return {
        language_column: base_frame[language_column].is_not_null().sum()
        for language_column in source_walkccc.LANGUAGE_COLUMNS
    }


def _print_solution_counts(solution_counts: dict[str, int]) -> None:
    for language_column, count in solution_counts.items():
        print(f"  {language_column:12s}: {count:,} solutions")


def _calculate_difficulty_distribution(base_frame: pl.DataFrame) -> pl.DataFrame:
    return base_frame["difficulty"].value_counts().sort("difficulty")


def _print_difficulty_distribution(difficulty_distribution: pl.DataFrame) -> None:
    print("\n  Difficulty distribution (from newfacade):")
    for row in difficulty_distribution.iter_rows(named=True):
        label = row["difficulty"] or "(unmatched)"
        print(f"    {label:10s}: {row['count']:,}")


def _calculate_input_output_coverage(base_frame: pl.DataFrame) -> tuple[int, int]:
    covered = base_frame["input_output"].is_not_null().sum()
    total = base_frame.height
    return covered, total


def _print_input_output_coverage(covered: int, total: int) -> None:
    print(
        f"  input_output: {covered:,} of {total:,} code snippets ({covered / total:.0%})"
    )


if __name__ == "__main__":
    main()
