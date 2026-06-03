from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import polars as pl

from llm_fine_tune.dataset import splits as splits_mod
from llm_fine_tune.dataset.instruction_generator import generate_instruction

BASE_PARQUET_PATH = Path("output/leetcode-solutions.parquet")
OUTPUT_DIR = Path("output")
INSTRUCT_TRAIN_PATH = OUTPUT_DIR / "leetcode-instruct-train.parquet"
INSTRUCT_TEST_PATH = OUTPUT_DIR / "leetcode-instruct-test.parquet"

INSTRUCT_LANGUAGES = ["cpp", "java", "python"]

DEFAULT_SEED = 0
DEFAULT_TEST_FRAC = 0.30
DEFAULT_SPLIT_SEED = 0


def _ensure_base_dataset() -> None:
    """Raise FileNotFoundError with an actionable message if the base parquet is missing."""
    if not BASE_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{BASE_PARQUET_PATH} not found — run `make base` first."
        )


def _load_base_dataframe() -> pl.DataFrame:
    """Read the base parquet into a Polars DataFrame."""
    return pl.read_parquet(BASE_PARQUET_PATH)


def _directed_language_pairs() -> list[tuple[str, str]]:
    """Return all ordered (source, target) pairs for the instruct languages."""
    return list(itertools.permutations(INSTRUCT_LANGUAGES, 2))


def _build_instruct_row(
    source_code: str,
    target_code: str,
    source_lang: str,
    target_lang: str,
    rng: random.Random,
    problem_id: int,
) -> dict:
    """Build a single instruct row tagged with its problem_id for split-key tracking."""
    return {
        "problem_id": problem_id,
        "instruction": generate_instruction(source_lang, target_lang, rng),
        "input": source_code,
        "output": target_code,
    }


def _collect_instruct_rows(df: pl.DataFrame, rng: random.Random) -> list[dict]:
    """Collect one row per (problem, directed language pair) where both solutions exist."""
    pairs = _directed_language_pairs()
    rows = []
    for problem in df.iter_rows(named=True):
        for source_lang, target_lang in pairs:
            source_code = problem[source_lang]
            target_code = problem[target_lang]
            if source_code is None or target_code is None:
                continue
            rows.append(
                _build_instruct_row(
                    source_code,
                    target_code,
                    source_lang,
                    target_lang,
                    rng,
                    problem["problem_id"],
                )
            )
    return rows


def _build_dataframe(rows: list[dict]) -> pl.DataFrame:
    """Convert collected rows into a typed Polars DataFrame (includes problem_id for splitting)."""
    schema = {
        "problem_id": pl.Int64,
        "instruction": pl.Utf8,
        "input": pl.Utf8,
        "output": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def _save_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write the DataFrame to path as a zstd-compressed Parquet file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")


def _print_summary(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    n_train_problems: int,
    n_test_problems: int,
) -> None:
    """Print per-split row and problem counts."""
    print(
        f"\nSaved instruct dataset:"
        f"\n  train: {train_df.height:,} rows ({n_train_problems:,} problems) → {INSTRUCT_TRAIN_PATH}"
        f"\n  test:  {test_df.height:,} rows ({n_test_problems:,} problems)  → {INSTRUCT_TEST_PATH}"
    )


def main() -> None:
    """Entry point. Loads the base dataset, generates instruct pairs, splits, and saves them."""
    parser = argparse.ArgumentParser(
        description="Build the leetcode-instruct Parquet dataset from the base dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for instruction template selection (default: %(default)s).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help="Fraction of problems held out for the test split (default: %(default)s).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed for the train/test split (default: %(default)s).",
    )
    args = parser.parse_args()

    _ensure_base_dataset()
    print(f"Loading base dataset from {BASE_PARQUET_PATH} ...")
    df = _load_base_dataframe()
    print(f"Generating instruct rows (seed={args.seed}) ...")
    rng = random.Random(args.seed)
    rows = _collect_instruct_rows(df, rng)
    print(f"Building dataset from {len(rows):,} rows ...")
    instruct_df = _build_dataframe(rows)

    print(
        f"Splitting by problem_id (test_frac={args.test_frac}, split_seed={args.split_seed}) ..."
    )
    train_df, test_df = splits_mod.split_by_key(
        instruct_df, "problem_id", args.test_frac, args.split_seed
    )

    n_train_problems = train_df["problem_id"].n_unique()
    n_test_problems = test_df["problem_id"].n_unique()

    train_df = train_df.drop("problem_id")
    test_df = test_df.drop("problem_id")

    _save_parquet(train_df, INSTRUCT_TRAIN_PATH)
    _save_parquet(test_df, INSTRUCT_TEST_PATH)
    _print_summary(train_df, test_df, n_train_problems, n_test_problems)


if __name__ == "__main__":
    main()
