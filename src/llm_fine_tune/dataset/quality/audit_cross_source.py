"""Cross-source agreement audit: walkccc Python vs newfacade completion.

For each Python row in the evaluation dataset, runs both the walkccc expected code snippet translation
(expected_code_snippet_translation) and the newfacade `completion` reference against
the shared typed test cases. Reports per-stratum agreement rates.

Run with: uv run audit-cross-source [--sample N | --all]
Requires: output/leetcode-evaluation.parquet (run `make evaluation` first).
Must run inside the execution harness Docker container.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import polars as pl
from tqdm import tqdm

from llm_fine_tune import loaders
from llm_fine_tune.execution_harness import execution
from llm_fine_tune.dataset import source_newfacade

EVALUATION_PATH = loaders.OUTPUT_DIR / "leetcode-evaluation.parquet"
REPORT_PATH = loaders.OUTPUT_DIR / "cross-source-audit.json"
DEFAULT_SAMPLE = 100
DEFAULT_SEED = 42


def main() -> None:
    args = _parse_args()
    loaders.require_file(EVALUATION_PATH, "run `make evaluation` first.")

    eval_frame = pl.read_parquet(EVALUATION_PATH).filter(
        pl.col("target_language") == "python"
    )

    newfacade_frame = source_newfacade.load_newfacade_frame()
    # newfacade uses problem_id as the join key; eval uses parallel_id
    joined = eval_frame.join(
        newfacade_frame.select(["problem_id", "completion"]),
        left_on="parallel_id",
        right_on="problem_id",
        how="left",
    )

    rows = list(joined.iter_rows(named=True))
    rng = random.Random(args.seed)
    if args.sample is not None:
        rng.shuffle(rows)
        rows = rows[: args.sample]

    results: list[dict] = []
    for row in tqdm(rows, desc="Auditing cross-source"):
        record = _check_row(row, args.timeout_s)
        results.append(record)

    _print_report(results)
    _write_report(results, REPORT_PATH)


def _check_row(row: dict, timeout_s: float) -> dict:
    expected_pairs = json.loads(row["expected_input_output_pairs"])
    engine = row["execution_engine"]

    walkccc_outcome = _run(
        engine,
        row["expected_code_snippet_translation"],
        "python",
        expected_pairs,
        timeout_s,
    )

    newfacade_completion = row.get("completion")
    if newfacade_completion:
        newfacade_outcome = _run(
            engine, newfacade_completion, "python", expected_pairs, timeout_s
        )
    else:
        newfacade_outcome = "no_completion"

    return {
        "parallel_id": row["parallel_id"],
        "difficulty": row.get("difficulty", "unknown"),
        "node_kind": _node_kind(engine),
        "walkccc": walkccc_outcome,
        "newfacade": newfacade_outcome,
        "agree": walkccc_outcome == newfacade_outcome,
    }


def _run(
    engine: str, code: str, language: str, expected_pairs: list[dict], timeout_s: float
) -> str:
    code_snippet_with_execution_wiring = (
        execution.assemble_code_snippet_with_execution_wiring(code, engine, language)
    )
    result = execution.compile_and_run(
        code_snippet_with_execution_wiring, language, timeout_s=timeout_s
    )
    produced = result["input_output_pairs_from_code_snippet"]
    if not result["compiled"]:
        return "compile_error"
    if len(produced) != len(expected_pairs):
        return "count_mismatch"
    if all(p["passed"] for p in produced):
        return "passed"
    return "failed_cases"


def _node_kind(execution_engine: str) -> str:
    if "TreeNode" in execution_engine:
        return "tree_node"
    if "ListNode" in execution_engine:
        return "list_node"
    return "plain"


def _print_report(results: list[dict]) -> None:
    total = len(results)
    agree_count = sum(1 for r in results if r["agree"])
    print(
        f"\nCross-source audit: {agree_count}/{total} rows agree ({agree_count / total * 100:.1f}%)\n"
    )

    by_stratum: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in results:
        by_stratum[(r["difficulty"], r["node_kind"])].append(r)

    print(f"{'difficulty':<14} {'node_kind':<12} {'agree':>7} {'total':>7} {'rate':>7}")
    print("-" * 52)
    for (diff, kind), rows in sorted(by_stratum.items()):
        n = len(rows)
        a = sum(1 for r in rows if r["agree"])
        print(f"{diff:<14} {kind:<12} {a:>7} {n:>7} {a / n * 100:>6.1f}%")

    walkccc_outcomes = collections.Counter(r["walkccc"] for r in results)
    newfacade_outcomes = collections.Counter(r["newfacade"] for r in results)
    print("\nwalkccc outcomes:", dict(walkccc_outcomes))
    print("newfacade outcomes:", dict(newfacade_outcomes))

    print(f"\nFull report written to: {REPORT_PATH}")


def _write_report(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit agreement between walkccc and newfacade Python reference solutions."
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        metavar="N",
        help=f"Random sample size (default: {DEFAULT_SAMPLE}).",
    )
    sample_group.add_argument(
        "--all",
        action="store_const",
        const=None,
        dest="sample",
        help="Run all rows.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout-s", type=float, default=10.0, dest="timeout_s")
    return parser.parse_args()


if __name__ == "__main__":
    main()
