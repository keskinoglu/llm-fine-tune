from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import polars as pl

SOURCE_REPO_URL = "https://github.com/walkccc/LeetCode"
SOURCE_REPO_DIR = Path("data/leetcode-source")
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


def _ensure_source_repo() -> None:
    if SOURCE_REPO_DIR.exists():
        print(f"Using existing clone at {SOURCE_REPO_DIR}")
        return
    print(f"Cloning {SOURCE_REPO_URL} into {SOURCE_REPO_DIR} ...")
    subprocess.run(
        ["git", "clone", SOURCE_REPO_URL, str(SOURCE_REPO_DIR)],
        check=True,
    )


def _update_source_repo() -> None:
    _ensure_source_repo()
    print(f"Pulling latest changes in {SOURCE_REPO_DIR} ...")
    subprocess.run(["git", "-C", str(SOURCE_REPO_DIR), "pull"], check=True)


def _parse_problem_folder(folder_name: str) -> tuple[int, str] | None:
    match = PROBLEM_FOLDER_PATTERN.match(folder_name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _read_solution_file(folder: Path, problem_id: int, extension: str) -> str | None:
    path = folder / f"{problem_id}.{extension}"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _build_problem_row(folder: Path, problem_id: int, title: str) -> dict:
    row: dict = {"problem_id": problem_id, "title": title}
    for extension, column in EXTENSION_TO_COLUMN.items():
        row[column] = _read_solution_file(folder, problem_id, extension)
    return row


def _collect_problem_rows() -> list[dict]:
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
    schema = {
        "problem_id": pl.Int64,
        "title": pl.Utf8,
        **{col: pl.Utf8 for col in LANGUAGE_COLUMNS},
    }
    return pl.DataFrame(rows, schema=schema).sort("problem_id")


def _save_parquet(df: pl.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH, compression="zstd")


def _print_summary(df: pl.DataFrame) -> None:
    print(f"\nSaved {df.height:,} problems to {OUTPUT_PATH}")
    for col in LANGUAGE_COLUMNS:
        count = df[col].is_not_null().sum()
        print(f"  {col:12s}: {count:,} solutions")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the leetcode-solutions Parquet dataset."
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull latest changes from the source repo before building.",
    )
    args = parser.parse_args()

    if args.pull:
        _update_source_repo()
    else:
        _ensure_source_repo()
    print("Scanning solutions...")
    rows = _collect_problem_rows()
    print(f"Building dataset from {len(rows):,} problems...")
    df = _build_dataframe(rows)
    _save_parquet(df)
    _print_summary(df)


if __name__ == "__main__":
    main()
