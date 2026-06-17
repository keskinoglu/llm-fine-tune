"""Phase 2: run each code_snippet_from_llm_response in its execution_engine and score it.

Standalone — no bigcode. Reads the Phase-1 generations and the evaluation parquet, pairs
them by row order, and writes the per-sample metrics JSON that evaluation-report consumes.
Runs inside the --net --network none container, so it loads the dataset from a local parquet (no Hub
access) and imports nothing heavier than polars + the execution harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from llm_fine_tune.evaluation import score
from llm_fine_tune.evaluation import extract_code_snippet_from_llm_response as extractor


def main() -> None:
    args = _parse_args()
    generations = json.loads(Path(args.generations_json).read_text())
    payloads = pl.read_parquet(args.evaluation_parquet).to_dicts()

    n = min(len(generations), len(payloads))
    records = []
    for i in range(n):
        payload = payloads[i]
        generation_list = generations[i]
        llm_response = generation_list[0] if generation_list else ""
        # Defensive: bigcode may save raw or already-postprocessed generations; extract
        # is idempotent on a bare code_snippet (no fence → returns it stripped).
        code_snippet_from_llm_response = (
            extractor.extract_code_snippet_from_llm_response(
                llm_response, payload["target_language"]
            )
        )
        records.append(
            score.score_bigcode_task_payload(payload, code_snippet_from_llm_response)
        )

    Path(args.metrics_json).write_text(json.dumps(records, indent=2))
    print(f"Scored {len(records)} payloads -> {args.metrics_json}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run + score Phase-1 generations against their execution_engines."
    )
    parser.add_argument(
        "--generations-json",
        required=True,
        help="Phase-1 output: list[list[str]] of generations, one inner list per payload.",
    )
    parser.add_argument(
        "--evaluation-parquet",
        required=True,
        help="The evaluation dataset rows (bigcode_task_payloads), same order as generation.",
    )
    parser.add_argument(
        "--metrics-json",
        required=True,
        help="Output path for the per-sample records consumed by evaluation-report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
