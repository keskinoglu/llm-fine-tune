"""Write per-sample evaluation results to Parquet and produce a summary markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from llm_fine_tune import loaders


def to_frame(records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(records)


def write_per_sample(records: list[dict], path: Path) -> None:
    loaders.write_parquet(to_frame(records), path)


def summarize(frame: pl.DataFrame) -> str:
    """Return a markdown table breaking down metrics by language pair and difficulty."""
    lines: list[str] = ["# Evaluation Summary\n"]

    group_cols = ["source_language", "target_language"]
    if "difficulty" in frame.columns:
        group_cols.append("difficulty")

    agg = (
        frame.group_by(group_cols)
        .agg(
            pl.len().alias("n"),
            pl.col("compiled").mean().alias("compile_rate"),
            pl.col("test_pass_rate").mean().alias("avg_pass_rate"),
            pl.col("pass@1").mean().alias("pass@1"),
        )
        .sort(group_cols)
    )

    lines.append(
        "| " + " | ".join(group_cols + ["n", "compile%", "avg_pass%", "pass@1"]) + " |"
    )
    lines.append("|" + "|".join(["---"] * (len(group_cols) + 4)) + "|")
    for row in agg.iter_rows(named=True):
        cells = [str(row[col]) for col in group_cols]
        cells += [
            str(row["n"]),
            f"{row['compile_rate']:.1%}",
            f"{row['avg_pass_rate']:.1%}",
            f"{row['pass@1']:.1%}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    records = json.loads(Path(args.results_json).read_text())
    frame = to_frame(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_per_sample(records, out_dir / "evaluation-results.parquet")
    summary = summarize(frame)
    (out_dir / "summary.md").write_text(summary)
    print(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation report from bigcode results JSON."
    )
    parser.add_argument(
        "--results-json", required=True, help="Path to bigcode metrics JSON."
    )
    parser.add_argument(
        "--out-dir", required=True, help="Directory for parquet + summary.md."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
