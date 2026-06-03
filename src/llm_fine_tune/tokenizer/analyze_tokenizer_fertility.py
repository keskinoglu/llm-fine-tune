"""Compute tokenizer fertility metrics across candidate models (Stage 2).

Tokenizes text samples with each model listed in tokenizer-sources.txt and
reports tokens-per-word, chars-per-token, and bytes-per-token. By default the
corpus is the full instruct dataset (train + test concatenated); pass --parquet
for a single file or --hf-dataset for a remote dataset. Results are printed to
stdout and saved to output/tokenizer-fertility-report.parquet.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import polars as pl
from datasets import load_dataset
from transformers import AutoTokenizer

from llm_fine_tune import loaders

DEFAULT_TOKENIZER_SOURCES_PATH = Path("tokenizer-sources.txt")
INSTRUCT_TRAIN_PATH = loaders.OUTPUT_DIR / "leetcode-instruct-train.parquet"
INSTRUCT_TEST_PATH = loaders.OUTPUT_DIR / "leetcode-instruct-test.parquet"
DEFAULT_INSTRUCT_PARQUETS = [INSTRUCT_TRAIN_PATH, INSTRUCT_TEST_PATH]
DEFAULT_TEXT_COLUMN = "input"
DEFAULT_HF_CONFIG = "instruct"
DEFAULT_HF_SPLIT = "train"
REPORT_PATH = loaders.OUTPUT_DIR / "tokenizer-fertility-report.parquet"

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def main() -> None:
    args = _parse_args()
    hf_token = os.getenv("HF_TOKEN")

    tokenizer_sources = _load_tokenizer_sources(args.tokenizer_sources)
    print(
        f"Loaded {len(tokenizer_sources)} tokenizer source(s) from {args.tokenizer_sources}"
    )

    texts = _load_texts(args, hf_token)
    if args.limit:
        texts = texts[: args.limit]
    print(f"Evaluating {len(texts):,} text samples.\n")

    fertility_rows = _evaluate_all_tokenizers(texts, tokenizer_sources, hf_token)

    if not fertility_rows:
        print("No results to report.")
        return

    report_frame = _build_report_frame(fertility_rows)
    loaders.write_parquet(report_frame, REPORT_PATH)
    _print_report(report_frame)


# ---- Argument parsing ----


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tokenizer fertility metrics for a list of HuggingFace model tokenizers."
    )
    parser.add_argument(
        "-s",
        "--tokenizer-sources",
        type=Path,
        default=DEFAULT_TOKENIZER_SOURCES_PATH,
        metavar="FILE",
        help="Path to tokenizer-sources.txt (default: %(default)s).",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_TEXT_COLUMN,
        help="Dataset column containing the text to tokenize (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N rows (useful for smoke tests).",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--parquet",
        type=Path,
        default=None,
        metavar="PATH",
        help="Evaluate a single local Parquet file instead of the full instruct dataset (train + test).",
    )
    input_group.add_argument(
        "--hf-dataset",
        metavar="NAME",
        help="HuggingFace dataset name (e.g. tkeskin/leetcode-solutions).",
    )
    parser.add_argument(
        "--hf-config",
        default=DEFAULT_HF_CONFIG,
        help="HuggingFace dataset config name (default: %(default)s).",
    )
    parser.add_argument(
        "--hf-split",
        default=DEFAULT_HF_SPLIT,
        help="HuggingFace dataset split (default: %(default)s).",
    )
    return parser.parse_args()


# ---- Text loading ----


def _load_texts(args: argparse.Namespace, hf_token: str | None) -> list[str]:
    if args.hf_dataset:
        print(
            f"Loading texts from HuggingFace dataset: "
            f"{args.hf_dataset} ({args.hf_config}/{args.hf_split}) ..."
        )
        return _load_texts_from_hf_dataset(
            args.hf_dataset, args.hf_config, args.hf_split, args.column, hf_token
        )
    if args.parquet:
        print(f"Loading texts from {args.parquet} ...")
        return _load_texts_from_parquet(args.parquet, args.column)
    print("Loading texts from the full instruct dataset (train + test) ...")
    return _concat_texts_from_parquets(DEFAULT_INSTRUCT_PARQUETS, args.column)


def _concat_texts_from_parquets(paths: list[Path], column: str) -> list[str]:
    texts: list[str] = []
    for path in paths:
        texts.extend(_load_texts_from_parquet(path, column))
    return texts


def _load_texts_from_parquet(path: Path, column: str) -> list[str]:
    loaders.require_file(path, "run `make instruct` first, or pass --hf-dataset.")
    parquet_frame = pl.read_parquet(path)
    if column not in parquet_frame.columns:
        raise ValueError(
            f"Column '{column}' not found in {path}. Available: {parquet_frame.columns}"
        )
    return parquet_frame[column].to_list()


def _load_texts_from_hf_dataset(
    name: str, config: str, split: str, column: str, hf_token: str | None
) -> list[str]:
    dataset = load_dataset(name, config, token=hf_token)[split]
    if column not in dataset.column_names:
        raise ValueError(
            f"Column '{column}' not found in {name}. Available: {dataset.column_names}"
        )
    return dataset[column]


# ---- Tokenizer evaluation ----


def _evaluate_all_tokenizers(
    texts: list[str],
    tokenizer_sources: dict[str, str],
    hf_token: str | None,
) -> list[dict]:
    results = []
    for display_name, hf_model_id in tokenizer_sources.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=hf_token)
            metrics = _calculate_fertility_metrics(texts, tokenizer)
            results.append(
                {"display_name": display_name, "hf_model_id": hf_model_id, **metrics}
            )
            print(
                f"  {display_name}: "
                f"tokens/word={metrics['tokens_per_word']:.3f}  "
                f"chars/token={metrics['chars_per_token']:.3f}  "
                f"bytes/token={metrics['bytes_per_token']:.3f}"
            )
        except Exception as error:
            print(f"  {display_name}: ERROR — {error}")
    return results


def _calculate_fertility_metrics(texts: list[str], tokenizer) -> dict[str, float]:
    r"""Corpus-level metrics (sum/sum, not per-sample average).

    Words are \w+ matches ([a-zA-Z0-9_]), so snake_case counts as one word
    and camelCase counts as one word.
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    for text in texts:
        total_tokens += len(tokenizer.tokenize(text))
        total_words += len(_WORD_RE.findall(text))
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
    return {
        "tokens_per_word": total_tokens / total_words,
        "chars_per_token": total_chars / total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "total_tokens": total_tokens,
        "total_words": total_words,
    }


# ---- Tokenizer source loading ----


def _load_tokenizer_sources(path: Path) -> dict[str, str]:
    loaders.require_file(path, "see README for the tokenizer-sources.txt format.")
    sources: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"Invalid line in {path}: {raw_line!r} — expected format: <display-name>=<hf-model-id>"
            )
        display_name, hf_model_id = line.split("=", 1)
        sources[display_name.strip()] = hf_model_id.strip()
    return sources


# ---- Reporting ----


def _build_report_frame(fertility_rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(fertility_rows).sort("tokens_per_word")


def _print_report(report_frame: pl.DataFrame) -> None:
    print(f"\nReport saved to {REPORT_PATH}")
    print("\nSorted by tokens/word (lower = better for code):")
    print(
        report_frame.select(
            ["display_name", "tokens_per_word", "chars_per_token", "bytes_per_token"]
        )
    )


if __name__ == "__main__":
    main()
