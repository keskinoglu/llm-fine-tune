"""Produce a base-vs-fine-tune comparison table from standard benchmark results.

Ingests, for each model result directory:
  perplexity.json               — from compute-heldout-perplexity
  lmeval/<model>/results_*.json — from lm_eval (nested, timestamped; located by glob)
  bigcode_<task>.json           — one per bigcode task (humaneval, multiple-cpp, -java, -py)

Emits benchmark-summary.md (base / fine-tune / delta table) + benchmark-results.parquet.

Usage:
  benchmark-report --base-dir <path> --ft-dir <path> [--out-dir <path>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from llm_fine_tune import loaders

_BIGCODE_TASKS = ["humaneval", "multiple-cpp", "multiple-java", "multiple-py"]

_LMEVAL_METRICS: dict[str, str] = {
    "mmlu": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "winogrande": "acc,none",
}


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _perplexity_value(result_dir: Path) -> float | None:
    data = _load_json(result_dir / "perplexity.json")
    if data:
        return data.get("perplexity")
    return None


def _lmeval_results_file(result_dir: Path) -> Path | None:
    # lm-eval with --log_samples treats --output_path as a directory and writes
    # <output_path>/<model_sanitized>/results_<timestamp>.json. Take the newest match.
    base = result_dir / "lmeval"
    candidates = sorted(base.rglob("results*.json")) if base.exists() else []
    return candidates[-1] if candidates else None


def _lmeval_values(result_dir: Path) -> dict[str, float | None]:
    results_file = _lmeval_results_file(result_dir)
    data = _load_json(results_file) if results_file else None
    if not data:
        return {task: None for task in _LMEVAL_METRICS}
    results = data.get("results", {})
    return {
        task: results.get(task, {}).get(metric)
        for task, metric in _LMEVAL_METRICS.items()
    }


def _bigcode_values(result_dir: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for task in _BIGCODE_TASKS:
        data = _load_json(result_dir / f"bigcode_{task}.json")
        if data is None:
            out[task] = None
            continue
        # bigcode writes {"pass@1": v} or {task: {"pass@1": v}}
        if "pass@1" in data:
            out[task] = data["pass@1"]
        else:
            nested = next(iter(data.values()), {})
            out[task] = nested.get("pass@1") if isinstance(nested, dict) else None
    return out


def _collect(result_dir: Path) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    row["perplexity"] = _perplexity_value(result_dir)
    row.update(_lmeval_values(result_dir))
    row.update(_bigcode_values(result_dir))
    return row


def _format(v: float | None, *, pct: bool = False) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    return f"{v:.2f}"


def _delta_str(
    base: float | None,
    ft: float | None,
    *,
    pct: bool = False,
    lower_is_better: bool = False,
) -> str:
    if base is None or ft is None:
        return "—"
    delta = ft - base
    if lower_is_better:
        delta = -delta
    sign = "+" if delta >= 0 else ""
    if pct:
        return f"{sign}{delta:.1%}"
    return f"{sign}{delta:.2f}"


def _build_summary(base_dir: Path, ft_dir: Path) -> tuple[str, pl.DataFrame]:
    base_model = _guess_model_name(base_dir)
    ft_model = _guess_model_name(ft_dir)

    base = _collect(base_dir)
    ft = _collect(ft_dir)

    # Define all metrics: (name, display_name, pct, lower_is_better)
    metrics = [
        ("perplexity", "perplexity (↓)", False, True),
        ("mmlu", "mmlu acc (↑)", True, False),
        ("gsm8k", "gsm8k (↑)", True, False),
        ("hellaswag", "hellaswag acc_norm (↑)", True, False),
        ("arc_challenge", "arc_challenge acc_norm (↑)", True, False),
        ("winogrande", "winogrande acc (↑)", True, False),
        ("humaneval", "HumanEval pass@1 (↑)", True, False),
        ("multiple-cpp", "MultiPL-E cpp pass@1 (↑)", True, False),
        ("multiple-java", "MultiPL-E java pass@1 (↑)", True, False),
        ("multiple-py", "MultiPL-E py pass@1 (↑)", True, False),
    ]

    header = f"| Metric | base ({base_model}) | fine-tune ({ft_model}) | Δ |\n"
    header += "|---|---|---|---|\n"

    rows_md = []
    rows_data = []
    for key, label, pct, lib in metrics:
        bv = base.get(key)
        fv = ft.get(key)
        rows_md.append(
            f"| {label} | {_format(bv, pct=pct)} | {_format(fv, pct=pct)} | {_delta_str(bv, fv, pct=pct, lower_is_better=lib)} |"
        )
        rows_data.append({"metric": label, "base": bv, "ft": fv})

    lines = ["# Standard Benchmark Comparison\n", header] + rows_md
    md = "\n".join(lines) + "\n"

    frame = pl.DataFrame(rows_data)
    return md, frame


def _guess_model_name(result_dir: Path) -> str:
    ppl = _load_json(result_dir / "perplexity.json")
    if ppl and ppl.get("model"):
        return ppl["model"].split("/")[-1]
    results_file = _lmeval_results_file(result_dir)
    lme = _load_json(results_file) if results_file else None
    if lme and lme.get("model_configs"):
        cfg = lme["model_configs"]
        if isinstance(cfg, dict):
            return str(cfg.get("pretrained", result_dir.name)).split("/")[-1]
    return result_dir.name


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir)
    ft_dir = Path(args.ft_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    md, frame = _build_summary(base_dir, ft_dir)

    summary_path = out_dir / "benchmark-summary.md"
    parquet_path = out_dir / "benchmark-results.parquet"
    summary_path.write_text(md)
    loaders.write_parquet(frame, parquet_path)
    print(md)
    print(f"Summary  -> {summary_path}")
    print(f"Parquet  -> {parquet_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce base-vs-fine-tune benchmark comparison."
    )
    parser.add_argument(
        "--base-dir", required=True, help="Result dir for the base model."
    )
    parser.add_argument(
        "--ft-dir", required=True, help="Result dir for the fine-tuned model."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write benchmark-summary.md + benchmark-results.parquet (default: cwd).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
