"""Phase 4 (code-benchmark track): assemble MultiPL-E completions + tests, run, compute pass@1.

Runs inside the --net --network none Apptainer sandbox (same image as run_execution_scoring.py).
Reads code_benchmark_generations.parquet, compiles+runs each row, writes a metrics JSON with
per-config pass@1 summary and per-sample records.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from llm_fine_tune.execution_harness import execution


def _assemble_multipl_e_program(completion: str, tests: str, language: str) -> str:
    preamble = execution.language_preamble(language)
    return preamble + "\n" + completion + "\n" + tests


def _score_row(row: dict) -> dict:
    program = _assemble_multipl_e_program(
        row["completion"], row["tests"], row["language"]
    )
    result = execution.compile_and_run_self_checking(program, row["language"])
    return {
        "config": row["config"],
        "name": row["name"],
        "language": row["language"],
        "compiled": result["compiled"],
        "passed": result["passed"],
        "diagnostics": result["diagnostics"][:2000],
    }


def main() -> None:
    args = _parse_args()
    rows = pl.read_parquet(args.generations).to_dicts()

    # Each row compiles+runs independently in its own tempdir (subprocess releases the GIL),
    # so threads parallelize cleanly across allocated CPUs. map preserves row order.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(_score_row, rows))

    by_config: dict[str, list[bool]] = {}
    for r in records:
        by_config.setdefault(r["config"], []).append(r["passed"])
    summary = {
        config: {"pass@1": sum(v) / len(v), "n": len(v)}
        for config, v in by_config.items()
    }

    Path(args.metrics_json).write_text(
        json.dumps({"summary": summary, "samples": records}, indent=2)
    )
    print(f"Scored {len(records)} rows -> {args.metrics_json}")
    for config, stats in summary.items():
        print(f"  {config}: pass@1 = {stats['pass@1']:.1%} ({stats['n']} samples)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MultiPL-E completions and compute pass@1 (Phase 4, code-benchmark track)."
    )
    parser.add_argument(
        "--generations",
        required=True,
        help="Path to code_benchmark_generations.parquet (Phase-3 output).",
    )
    parser.add_argument(
        "--metrics-json",
        required=True,
        help="Output path for metrics JSON consumed by benchmark-report.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Parallel compile+run workers (default: allocated CPUs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
