"""Phase 4 (code-benchmark track): assemble MultiPL-E completions + tests, run, compute pass@1.

Runs inside the --net --network none Apptainer sandbox (same image as run_execution_scoring.py).
Reads code_benchmark_generations.parquet, compiles+runs each row, writes a metrics JSON with
per-config pass@1 summary and per-sample records.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from llm_fine_tune.execution_harness import execution


_JAVA_CLASS_NAME_RE = re.compile(r"\bclass\s+(\w+)")
_JAVA_CLASS_OPEN_RE = re.compile(r"\bclass\s+\w+[^{]*\{", re.DOTALL)


def _function_body(completion: str) -> str:
    """Return the body between the outermost braces of a complete C++ function.

    MultiPL-E's C++ prompt opens the free-function signature (ends with `... {`) and its tests begin
    with the matching `}`. The instruct-wrapped model returns a *whole* function, so we drop its
    signature and outer braces and slot just the body into MultiPL-E's own signature — robust to the
    model renaming the function, since the tests call the name MultiPL-E baked into the prompt.
    """
    open_i = completion.find("{")
    close_i = completion.rfind("}")
    if open_i == -1 or close_i == -1 or close_i <= open_i:
        return completion
    return completion[open_i + 1 : close_i]


def _java_method_source(completion: str) -> str:
    """Extract the method(s) from a Java completion.

    Java has no free functions, so an instruct model answers "complete this function" with either a
    bare method or a whole `class Foo { ... }`. MultiPL-E wants the method(s) inside its own Problem
    class, so when the model wraps them in a class we take the class body; otherwise the completion is
    already bare. Unlike C++ we keep the full signature (a returned class nests the method, so the
    "first { to last }" body trick would capture the signature too and duplicate it — the bug that
    scored java ~0 for both models).
    """
    m = _JAVA_CLASS_OPEN_RE.search(completion)
    if not m:
        return completion.strip()
    body_end = completion.rfind("}")
    if body_end <= m.end():
        return completion.strip()
    return completion[m.end() : body_end].strip()


def _java_main_block(tests: str) -> str:
    """MultiPL-E's java tests are raw-completion shaped: a leading `}` closes the prompt's open
    method and a trailing `}` closes its class. Strip both to leave the bare `main(...) { ... }`."""
    t = tests.strip()
    if t.startswith("}"):
        t = t[1:].lstrip()
    if t.endswith("}"):
        t = t[:-1].rstrip()
    return t


def _assemble_java(prompt: str, completion: str, tests: str) -> str:
    """Rebuild a single-class Java program: the prompt's imports + a Problem class wrapping the
    model's method(s) and MultiPL-E's main(). Handles the model returning either a bare method or a
    full class without duplicating the method signature."""
    imports = "\n".join(
        line for line in prompt.splitlines() if line.lstrip().startswith("import")
    )
    name_match = _JAVA_CLASS_NAME_RE.search(prompt)
    class_name = name_match.group(1) if name_match else "Problem"
    methods = _java_method_source(completion)
    main = _java_main_block(tests)
    return f"{imports}\nclass {class_name} {{\n{methods}\n{main}\n}}\n"


def _assemble_multipl_e_program(
    completion: str, tests: str, language: str, prompt: str
) -> str:
    if language == "java":
        return _assemble_java(prompt, completion, tests)
    if language == "cpp":
        return prompt + "\n" + _function_body(completion) + "\n" + tests
    # Python comes from canonical HumanEval: the model returns a full function that the appended
    # check(<entry_point>) harness calls by name, so use it whole under the stdlib preamble.
    preamble = execution.language_preamble(language)
    return preamble + "\n" + completion + "\n" + tests


def _score_row(row: dict) -> dict:
    program = _assemble_multipl_e_program(
        row["completion"], row["tests"], row["language"], row["prompt"]
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
