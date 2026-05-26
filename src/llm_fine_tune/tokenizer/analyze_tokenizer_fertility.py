from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import polars as pl
from datasets import load_dataset
from transformers import AutoTokenizer

OUTPUT_DIR = Path("output")

DEFAULT_TOKENIZER_SOURCES_PATH = Path("tokenizer-sources.txt")
DEFAULT_INPUT_PARQUET = OUTPUT_DIR / "leetcode-instruct.parquet"
DEFAULT_TEXT_COLUMN = "input"
DEFAULT_HF_CONFIG = "instruct"
DEFAULT_HF_SPLIT = "train"
REPORT_PATH = OUTPUT_DIR / "tokenizer-fertility-report.parquet"

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _load_tokenizer_sources(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — see README for the tokenizer-sources.txt format."
        )
    sources: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"Invalid line in {path}: {raw!r} — expected format: <display-name>=<hf-model-id>"
            )
        name, hf_model_id = line.split("=", 1)
        sources[name.strip()] = hf_model_id.strip()
    return sources


def _load_texts_from_parquet(path: Path, column: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `make instruct` first, or pass --hf-dataset."
        )
    df = pl.read_parquet(path)
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {path}. Available: {df.columns}"
        )
    return df[column].to_list()


def _load_texts_from_hf_dataset(
    name: str, config: str, split: str, column: str, hf_token: str | None
) -> list[str]:
    dataset = load_dataset(name, config, token=hf_token)[split]
    if column not in dataset.column_names:
        raise ValueError(
            f"Column '{column}' not found in {name}. Available: {dataset.column_names}"
        )
    return dataset[column]


def _metrics_for_tokenizer(texts: list[str], tokenizer) -> dict[str, float]:
    r"""All metrics are corpus-level (sum/sum, not per-sample average). Words are \w+ matches
    ([a-zA-Z0-9_]), so snake_case counts as one word and camelCase counts as one word."""
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


def _build_report_dataframe(results: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(results).sort("tokens_per_word")


def _save_report(df: pl.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(REPORT_PATH, compression="zstd")


def main() -> None:
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
        default=DEFAULT_INPUT_PARQUET,
        metavar="PATH",
        help="Path to a local Parquet file (default: %(default)s).",
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

    args = parser.parse_args()
    hf_token = os.getenv("HF_TOKEN")

    tokenizer_sources = _load_tokenizer_sources(args.tokenizer_sources)
    print(
        f"Loaded {len(tokenizer_sources)} tokenizer source(s) from {args.tokenizer_sources}"
    )

    if args.hf_dataset:
        print(
            f"Loading texts from HuggingFace dataset: {args.hf_dataset} ({args.hf_config}/{args.hf_split}) ..."
        )
        texts = _load_texts_from_hf_dataset(
            args.hf_dataset, args.hf_config, args.hf_split, args.column, hf_token
        )
    else:
        print(f"Loading texts from {args.parquet} ...")
        texts = _load_texts_from_parquet(args.parquet, args.column)

    if args.limit:
        texts = texts[: args.limit]

    print(f"Evaluating {len(texts):,} text samples.\n")

    results = []
    for display_name, hf_model_id in tokenizer_sources.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=hf_token)
            metrics = _metrics_for_tokenizer(texts, tokenizer)
            results.append(
                {"display_name": display_name, "hf_model_id": hf_model_id, **metrics}
            )
            print(
                f"  {display_name}: "
                f"tokens/word={metrics['tokens_per_word']:.3f}  "
                f"chars/token={metrics['chars_per_token']:.3f}  "
                f"bytes/token={metrics['bytes_per_token']:.3f}"
            )
        except Exception as e:
            print(f"  {display_name}: ERROR — {e}")

    if not results:
        print("No results to report.")
        return

    df = _build_report_dataframe(results)
    _save_report(df)
    print(f"\nReport saved to {REPORT_PATH}")
    print("\nSorted by tokens/word (lower = better for code):")
    print(
        df.select(
            ["display_name", "tokens_per_word", "chars_per_token", "bytes_per_token"]
        )
    )


if __name__ == "__main__":
    main()
