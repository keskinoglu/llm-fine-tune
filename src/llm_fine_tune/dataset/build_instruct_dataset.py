"""Build the instruct Parquet datasets for fine-tuning (Stage 1, step 2).

Reads output/leetcode-solutions.parquet, expands each problem into directed
code-translation pairs across C++, Java, and Python, splits them 70/30 by
problem, then writes output/leetcode-instruct-train.parquet and
output/leetcode-instruct-test.parquet.
"""

from __future__ import annotations

import argparse
import itertools
import random

import polars as pl

from llm_fine_tune import loaders
from llm_fine_tune.dataset import splits
from llm_fine_tune.dataset.instruction_generator import generate_instruction

BASE_PARQUET_PATH = loaders.OUTPUT_DIR / "leetcode-solutions.parquet"
INSTRUCT_TRAIN_PATH = loaders.OUTPUT_DIR / "leetcode-instruct-train.parquet"
INSTRUCT_TEST_PATH = loaders.OUTPUT_DIR / "leetcode-instruct-test.parquet"

INSTRUCT_LANGUAGES = ["cpp", "java", "python"]

DEFAULT_SEED = 0
DEFAULT_TEST_FRAC = 0.30
DEFAULT_SPLIT_SEED = 0


def main() -> None:
    args = _parse_args()

    loaders.require_file(BASE_PARQUET_PATH, "run `make base` first.")
    print(f"Loading base dataset from {BASE_PARQUET_PATH} ...")
    base_frame = pl.read_parquet(BASE_PARQUET_PATH)

    instruction_rng = random.Random(args.seed)
    print(f"Generating instruct rows (seed={args.seed}) ...")
    instruct_rows = _collect_instruct_rows(base_frame, instruction_rng)

    print(f"Building dataset from {len(instruct_rows):,} rows ...")
    instruct_frame = _build_instruct_frame(instruct_rows)

    print(
        f"Splitting by code_snippet (test_frac={args.test_frac}, split_seed={args.split_seed}) ..."
    )
    train_frame, test_frame = splits.split_by_key(
        instruct_frame, "parallel_id", args.test_frac, args.split_seed
    )

    n_train_snippets = train_frame["parallel_id"].n_unique()
    n_test_snippets = test_frame["parallel_id"].n_unique()

    train_frame = train_frame.drop("parallel_id")
    test_frame = test_frame.drop("parallel_id")

    loaders.write_parquet(train_frame, INSTRUCT_TRAIN_PATH)
    loaders.write_parquet(test_frame, INSTRUCT_TEST_PATH)
    _print_summary(train_frame, test_frame, n_train_snippets, n_test_snippets)


# ---- Argument parsing ----


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the leetcode-instruct Parquet datasets from the base dataset."
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
    return parser.parse_args()


# ---- Row generation ----


def _collect_instruct_rows(
    base_frame: pl.DataFrame, instruction_rng: random.Random
) -> list[dict]:
    directed_pairs = list(itertools.permutations(INSTRUCT_LANGUAGES, 2))
    rows = []
    for code_snippet in base_frame.iter_rows(named=True):
        for source_language, target_language in directed_pairs:
            source_code = code_snippet[source_language]
            target_code = code_snippet[target_language]
            if source_code is None or target_code is None:
                continue
            rows.append(
                _build_instruct_row(
                    parallel_id=code_snippet["parallel_id"],
                    source_language=source_language,
                    target_language=target_language,
                    source_code=source_code,
                    target_code=target_code,
                    instruction_rng=instruction_rng,
                )
            )
    return rows


def _build_instruct_row(
    *,
    parallel_id: int,
    source_language: str,
    target_language: str,
    source_code: str,
    target_code: str,
    instruction_rng: random.Random,
) -> dict:
    return {
        "parallel_id": parallel_id,
        "instruction": generate_instruction(
            source_language, target_language, instruction_rng
        ),
        "input": source_code,
        "output": target_code,
    }


def _build_instruct_frame(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "parallel_id": pl.Int64,
        "instruction": pl.Utf8,
        "input": pl.Utf8,
        "output": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


# ---- Summary ----


def _print_summary(
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    n_train_snippets: int,
    n_test_snippets: int,
) -> None:
    print(
        f"\nSaved instruct dataset:"
        f"\n  train: {train_frame.height:,} rows ({n_train_snippets:,} code snippets) → {INSTRUCT_TRAIN_PATH}"
        f"\n  test:  {test_frame.height:,} rows ({n_test_snippets:,} code snippets)  → {INSTRUCT_TEST_PATH}"
    )


if __name__ == "__main__":
    main()
