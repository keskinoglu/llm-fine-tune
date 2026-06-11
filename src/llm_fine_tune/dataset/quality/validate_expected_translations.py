"""Validate expected code snippet translations for dataset quality auditing.

For each row in the evaluation dataset, runs the expected code snippet translation through its
execution engine and classifies the outcome. Non-passing rows are written to
dataset/quality/exclusions.json (a committed artifact consumed by build_evaluation_dataset.py).

Run with: uv run validate-expected-translations [--target-language {python,cpp,java}] [--sample N | --all]
Requires: output/leetcode-evaluation.parquet (run `make evaluation` first).
Must run inside the execution harness Docker container (--network=none, correct runtimes present).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm import tqdm

from llm_fine_tune import loaders
from llm_fine_tune.execution_harness import execution

EVALUATION_PATH = loaders.OUTPUT_DIR / "leetcode-evaluation.parquet"
EXCLUSIONS_PATH = Path(__file__).parent / "exclusions.json"
FAILURES_PATH = loaders.OUTPUT_DIR / "expected-translation-failures.json"
DEFAULT_SAMPLE = 50
DEFAULT_SEED = 42


def main() -> None:
    args = _parse_args()
    loaders.require_file(EVALUATION_PATH, "run `make evaluation` first.")
    frame = pl.read_parquet(EVALUATION_PATH)
    rows = _select_rows(frame, args.target_language or None, args.sample, args.seed)

    results, failures = _validate(rows, args.timeout_s, args.workers, args.fail_fast)

    _print_report(results, args.show_failures)

    if args.write_exclusions:
        _write_exclusions(failures)
        print(f"Exclusions written to: {EXCLUSIONS_PATH}")

    _write_failures(failures, FAILURES_PATH)


def _validate(
    rows: list[dict], timeout_s: float, workers: int, fail_fast: bool
) -> tuple[list[dict], list[dict]]:
    """Run check_row over every row, collecting results and non-passing failures.

    Each row's expected code snippet translation is compiled and run
    independently, so the work parallelises cleanly across threads
    (subprocess.run releases the GIL; each execute() gets its own tempdir).
    --fail-fast forces sequential order so the first failure is deterministic.
    """
    results: list[dict] = []
    failures: list[dict] = []

    if fail_fast:
        for row in tqdm(rows, desc="Validating expected translations"):
            record = check_row(row, timeout_s)
            results.append(record)
            if record["outcome"] != "passed":
                failures.append(record)
                break
        return results, failures

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(check_row, row, timeout_s) for row in rows]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Validating expected translations",
        ):
            record = future.result()
            results.append(record)
            if record["outcome"] != "passed":
                failures.append(record)

    return results, failures


def check_row(row: dict, timeout_s: float) -> dict:
    """Build the expected code snippet translation + its engine, run it, classify the outcome."""
    target_language = row["target_language"]
    expected_input_output_pairs = json.loads(row["expected_input_output_pairs"])
    executable = execution.build_executable_code_snippet_from_llm_response(
        row["execution_engine"],
        row["expected_code_snippet_translation"],
        target_language,
    )
    result = execution.execute(executable, target_language, timeout_s=timeout_s)

    produced = result["input_output_pairs_from_llm_generated_code"]
    expected_n = len(expected_input_output_pairs)
    if not result["compiled"]:
        outcome = "compile_error"
    elif len(produced) != expected_n:
        outcome = "count_mismatch"
    elif all(pair["passed"] for pair in produced):
        outcome = "passed"
    else:
        outcome = "failed_cases"

    record: dict = {
        "parallel_id": row["parallel_id"],
        "target_language": target_language,
        "difficulty": row.get("difficulty", "unknown"),
        "node_kind": _node_kind(row["execution_engine"]),
        "outcome": outcome,
        "expected_n": expected_n,
        "observed_n": len(produced),
    }
    if outcome != "passed":
        record["diagnostics"] = result["diagnostics"][:2000]
        record["executable"] = executable
    return record


def _node_kind(execution_engine: str) -> str:
    if "TreeNode" in execution_engine:
        return "tree_node"
    if "ListNode" in execution_engine:
        return "list_node"
    return "plain"


def _select_rows(
    frame: pl.DataFrame,
    target_languages: list[str] | None,
    sample: int | None,
    seed: int,
) -> list[dict]:
    if target_languages:
        frame = frame.filter(pl.col("target_language").is_in(target_languages))
    # Validation depends only on the target side: every source-language variant
    # of a (parallel_id, target_language) carries the identical
    # expected_code_snippet_translation, execution_engine, and
    # expected_input_output_pairs. Collapse to one row per pair — the
    # granularity the exclusion list is keyed by.
    frame = frame.unique(
        subset=["parallel_id", "target_language"], keep="first", maintain_order=True
    )
    all_rows = list(frame.iter_rows(named=True))
    if sample is None:
        return all_rows
    rng = random.Random(seed)
    by_stratum: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in all_rows:
        key = (
            row["target_language"],
            row.get("difficulty", "unknown"),
            _node_kind(row["execution_engine"]),
        )
        by_stratum[key].append(row)
    for rows in by_stratum.values():
        rng.shuffle(rows)
    n_strata = len(by_stratum)
    base = max(1, sample // n_strata)
    selected = []
    remainder: list[dict] = []
    for rows in by_stratum.values():
        selected.extend(rows[:base])
        remainder.extend(rows[base:])
    rng.shuffle(remainder)
    needed = sample - len(selected)
    if needed > 0:
        selected.extend(remainder[:needed])
    return selected


def _print_report(results: list[dict], show_failures: bool) -> None:
    outcomes = collections.Counter(r["outcome"] for r in results)
    total = len(results)
    passed = outcomes.get("passed", 0)
    print(
        f"\nExpected translation validation: {passed}/{total} passed ({passed / total * 100:.1f}%)\n"
    )

    by_lang_kind: dict[tuple, list[str]] = collections.defaultdict(list)
    for r in results:
        by_lang_kind[(r["target_language"], r["node_kind"])].append(r["outcome"])
    print(f"{'language':<10} {'node_kind':<12} {'passed':>7} {'total':>7} {'rate':>7}")
    print("-" * 48)
    for (lang, kind), oc in sorted(by_lang_kind.items()):
        n = len(oc)
        p = sum(1 for o in oc if o == "passed")
        print(f"{lang:<10} {kind:<12} {p:>7} {n:>7} {p / n * 100:>6.1f}%")

    if any(k != "passed" for k in outcomes):
        print("\nFailures by outcome:")
        for outcome, n in sorted(outcomes.items()):
            if outcome != "passed":
                print(f"  {outcome:<20} {n:>5}")

    if show_failures:
        for f in results:
            if f["outcome"] != "passed":
                print(
                    f"\n--- FAIL parallel_id={f['parallel_id']} lang={f['target_language']} "
                    f"node_kind={f['node_kind']} outcome={f['outcome']} "
                    f"expected_n={f['expected_n']} observed_n={f['observed_n']}"
                )
                if f.get("diagnostics"):
                    print("diagnostics:", f["diagnostics"][:500])

    print(f"\nFailures written to: {FAILURES_PATH}")


def _write_exclusions(failures: list[dict]) -> None:
    existing: list[dict] = []
    if EXCLUSIONS_PATH.exists():
        existing = json.loads(EXCLUSIONS_PATH.read_text())

    existing_keys = {(r["parallel_id"], r["target_language"]) for r in existing}
    new_records = [
        {
            "parallel_id": f["parallel_id"],
            "target_language": f["target_language"],
            "outcome": f["outcome"],
        }
        for f in failures
        if (f["parallel_id"], f["target_language"]) not in existing_keys
    ]
    merged = existing + new_records
    EXCLUSIONS_PATH.write_text(json.dumps(merged, indent=2))


def _write_failures(failures: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(failures, fh, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate expected code snippet translations against typed test-column input output pairs."
    )
    parser.add_argument(
        "--target-language",
        action="append",
        choices=["python", "cpp", "java"],
        dest="target_language",
        help="Language(s) to check (repeatable). Defaults to all.",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        metavar="N",
        help=f"Stratified sample size (default: {DEFAULT_SAMPLE}).",
    )
    sample_group.add_argument(
        "--all",
        action="store_const",
        const=None,
        dest="sample",
        help="Run all rows (overrides --sample).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout-s", type=float, default=10.0, dest="timeout_s")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        dest="workers",
        help="Parallel compile/run workers (default: CPU count = %(default)s). "
        "The wrapper passes the container CPU budget.",
    )
    parser.add_argument("--fail-fast", action="store_true", dest="fail_fast")
    parser.add_argument("--show-failures", action="store_true", dest="show_failures")
    parser.add_argument(
        "--write-exclusions",
        action="store_true",
        dest="write_exclusions",
        help=f"Merge non-passing rows into {EXCLUSIONS_PATH} (committed artifact).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
