from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import polars as pl

from llm_fine_tune.instruction_generator import generate_instruction

BASE_PARQUET_PATH = Path("output/leetcode-solutions.parquet")
OUTPUT_DIR = Path("output")
INSTRUCT_OUTPUT_PATH = OUTPUT_DIR / "leetcode-instruct.parquet"

INSTRUCT_LANGUAGES = ["cpp", "java", "python"]

DEFAULT_SEED = 0


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
) -> dict:
    """Build a single instruct row with instruction, input, and output."""
    return {
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
                    source_code, target_code, source_lang, target_lang, rng
                )
            )
    return rows


def _build_dataframe(rows: list[dict]) -> pl.DataFrame:
    """Convert collected rows into a typed Polars DataFrame."""
    schema = {"instruction": pl.Utf8, "input": pl.Utf8, "output": pl.Utf8}
    return pl.DataFrame(rows, schema=schema)


def _save_parquet(df: pl.DataFrame) -> None:
    """Write the DataFrame to INSTRUCT_OUTPUT_PATH as a zstd-compressed Parquet file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(INSTRUCT_OUTPUT_PATH, compression="zstd")


def _print_summary(df: pl.DataFrame) -> None:
    """Print total row count to stdout."""
    print(f"\nSaved {df.height:,} instruct rows to {INSTRUCT_OUTPUT_PATH}")


def main() -> None:
    """Entry point. Loads the base dataset, generates instruct pairs, and saves them."""
    parser = argparse.ArgumentParser(
        description="Build the leetcode-instruct Parquet dataset from the base dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for instruction template selection (default: %(default)s).",
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
    _save_parquet(instruct_df)
    _print_summary(instruct_df)


if __name__ == "__main__":
    main()
