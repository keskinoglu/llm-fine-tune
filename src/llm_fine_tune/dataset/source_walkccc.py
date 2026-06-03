"""Schema adapter for the walkccc/LeetCode GitHub repository.

Clones or updates the repository locally via source_cache, then walks the
solutions directory and assembles one row per LeetCode problem — with one
column per supported language (null when no solution exists for that problem).
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from llm_fine_tune.dataset import source_cache

SOURCE_REPO_URL = "https://github.com/walkccc/LeetCode"
SOURCE_REPO_DIR = source_cache.DATA_DIR / "leetcode-source"
SOLUTIONS_DIR = SOURCE_REPO_DIR / "solutions"

EXTENSION_TO_LANGUAGE_COLUMN: dict[str, str] = {
    "cpp": "cpp",
    "java": "java",
    "py": "python",
    "sql": "sql",
    "ts": "typescript",
}

LANGUAGE_COLUMNS: list[str] = list(EXTENSION_TO_LANGUAGE_COLUMN.values())

_PROBLEM_FOLDER_PATTERN = re.compile(r"^(\d+)\. (.+)$")


def load_walkccc_frame(*, pull: bool = False) -> pl.DataFrame:
    """Return a problem_id-keyed frame with one column per language solution.

    Clones the source repository on first call; pass pull=True to update it.
    """
    source_cache.ensure_git_repo(SOURCE_REPO_URL, SOURCE_REPO_DIR, update=pull)
    print("Scanning solutions ...")
    rows = _collect_problem_rows()
    print(f"Building dataset from {len(rows):,} problems ...")
    return _build_walkccc_frame(rows)


# ---- Internal helpers (scan → row per problem → typed frame) ----


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


def _parse_problem_folder(folder_name: str) -> tuple[int, str] | None:
    match = _PROBLEM_FOLDER_PATTERN.match(folder_name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _build_problem_row(folder: Path, problem_id: int, title: str) -> dict:
    row: dict = {"problem_id": problem_id, "title": title}
    for extension, language_column in EXTENSION_TO_LANGUAGE_COLUMN.items():
        row[language_column] = _read_solution_file(folder, problem_id, extension)
    return row


def _read_solution_file(folder: Path, problem_id: int, extension: str) -> str | None:
    path = folder / f"{problem_id}.{extension}"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _build_walkccc_frame(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "problem_id": pl.Int64,
        "title": pl.Utf8,
        **{column: pl.Utf8 for column in LANGUAGE_COLUMNS},
    }
    return pl.DataFrame(rows, schema=schema).sort("problem_id")
