from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from llm_fine_tune.dataset import newfacade_source, sources

SOURCE_REPO_URL = "https://github.com/walkccc/LeetCode"
SOURCE_REPO_DIR = sources.DATA_DIR / "leetcode-source"
SOLUTIONS_DIR = SOURCE_REPO_DIR / "solutions"
OUTPUT_DIR = Path("output")
OUTPUT_PATH = OUTPUT_DIR / "leetcode-solutions.parquet"

EXTENSION_TO_COLUMN: dict[str, str] = {
    "cpp": "cpp",
    "java": "java",
    "py": "python",
    "sql": "sql",
    "ts": "typescript",
}

LANGUAGE_COLUMNS = list(EXTENSION_TO_COLUMN.values())

PROBLEM_FOLDER_PATTERN = re.compile(r"^(\d+)\. (.+)$")


def _parse_problem_folder(folder_name: str) -> tuple[int, str] | None:
    """Parse a folder name like '1. Two Sum' into (1, 'Two Sum'). Returns None if it doesn't match."""
    match = PROBLEM_FOLDER_PATTERN.match(folder_name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _read_solution_file(folder: Path, problem_id: int, extension: str) -> str | None:
    """Read a solution file for the given problem and extension. Returns None if absent."""
    path = folder / f"{problem_id}.{extension}"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _build_problem_row(folder: Path, problem_id: int, title: str) -> dict:
    """Build a single dataset row with metadata and all available language solutions."""
    row: dict = {"problem_id": problem_id, "title": title}
    for extension, column in EXTENSION_TO_COLUMN.items():
        row[column] = _read_solution_file(folder, problem_id, extension)
    return row


def _collect_problem_rows() -> list[dict]:
    """Walk the solutions directory and collect one row per problem."""
    rows = []
    for folder in sorted(SOLUTIONS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        parsed = _parse_problem_folder(folder.name)
        if parsed is None:
            print(f"  Skipping unexpected folder: {folder.name}")
            continue
        problem_id, title = parsed
        rows.append(_build_problem_row(folder, problem_id, title))
    return rows


def _build_dataframe(rows: list[dict]) -> pl.DataFrame:
    """Convert collected rows into a typed, sorted Polars DataFrame."""
    schema = {
        "problem_id": pl.Int64,
        "title": pl.Utf8,
        **{col: pl.Utf8 for col in LANGUAGE_COLUMNS},
    }
    return pl.DataFrame(rows, schema=schema).sort("problem_id")


def _enrich(base_df: pl.DataFrame, nf_df: pl.DataFrame) -> pl.DataFrame:
    """Left-join newfacade columns into base on problem_id. Null where unmatched."""
    return base_df.join(nf_df, on="problem_id", how="left")


def _print_integrity_report(base_df: pl.DataFrame, nf_df: pl.DataFrame) -> None:
    """Print a coverage + title-match report; thin shell over pure helpers."""
    base_ids = set(base_df["problem_id"].to_list())
    nf_ids = set(nf_df["problem_id"].to_list())
    in_nf_only, in_base_only = newfacade_source.unmatched_problem_ids(base_ids, nf_ids)
    mismatches = newfacade_source.title_mismatches(base_df, nf_df)

    print(
        f"\nIntegrity check: {len(base_ids):,} base problems, "
        f"{len(nf_ids):,} newfacade problems"
    )
    print(f"  Matched:           {len(base_ids & nf_ids):,}")
    ids_preview = sorted(in_nf_only)[:10]
    ellipsis = "..." if len(in_nf_only) > 10 else ""
    print(
        f"  In newfacade only: {len(in_nf_only):,}"
        + (f" — ids: {ids_preview}{ellipsis}" if in_nf_only else "")
    )
    print(f"  In base only:      {len(in_base_only):,}")
    if mismatches:
        print(f"  Title mismatches:  {len(mismatches):,}")
        for m in mismatches[:5]:
            print(
                f"    id={m['problem_id']}: "
                f"'{m['base_title']}' vs task_id='{m['task_id']}'"
            )
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches) - 5} more")
    else:
        print("  Title mismatches:  0")


def _save_parquet(df: pl.DataFrame) -> None:
    """Write the DataFrame to OUTPUT_PATH as a zstd-compressed Parquet file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH, compression="zstd")


def _print_summary(df: pl.DataFrame) -> None:
    """Print row count and per-language solution coverage to stdout."""
    print(f"\nSaved {df.height:,} problems to {OUTPUT_PATH}")
    for col in LANGUAGE_COLUMNS:
        count = df[col].is_not_null().sum()
        print(f"  {col:12s}: {count:,} solutions")
    if "difficulty" in df.columns:
        print("\n  Difficulty distribution (from newfacade):")
        counts = df["difficulty"].value_counts().sort("difficulty")
        for row in counts.iter_rows(named=True):
            label = row["difficulty"] or "(unmatched)"
            print(f"    {label:10s}: {row['count']:,}")
    if "input_output" in df.columns:
        io_count = df["input_output"].is_not_null().sum()
        print(
            f"  input_output: {io_count:,} of {df.height:,} problems ({io_count / df.height:.0%})"
        )


def main() -> None:
    """Entry point. Ensures external sources are available, builds the dataset, and saves it."""
    parser = argparse.ArgumentParser(
        description="Build the leetcode-solutions Parquet dataset."
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull latest changes from the walkccc/LeetCode source repo before building.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download all cached HF datasets (e.g. newfacade) before building.",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip newfacade/LeetCodeDataset enrichment (offline or debug mode).",
    )
    args = parser.parse_args()

    sources.ensure_git_repo(SOURCE_REPO_URL, SOURCE_REPO_DIR, update=args.pull)
    print("Scanning solutions...")
    rows = _collect_problem_rows()
    print(f"Building dataset from {len(rows):,} problems...")
    df = _build_dataframe(rows)

    if not args.no_enrich:
        nf_df = newfacade_source.load_newfacade_frame(refresh=args.refresh)
        _print_integrity_report(df, nf_df)
        df = _enrich(df, nf_df)

    _save_parquet(df)
    _print_summary(df)


if __name__ == "__main__":
    main()
