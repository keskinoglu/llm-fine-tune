"""Metric measures for one code_snippet_translation evaluation sample.

Add a new metric by adding a function here and calling it from score.py.
"""

from __future__ import annotations

from llm_fine_tune.execution_harness.execution import ExecutionResult


def measure_compilation_success(execution_result: ExecutionResult) -> float:
    return 1.0 if execution_result["compiled"] else 0.0


def measure_correctness(
    execution_result: ExecutionResult,
    expected_input_output_pairs: list[dict],
) -> float:
    """Fraction of test cases where the generated code's output matches expected."""
    generated = execution_result["input_output_pairs_from_code_snippet"]
    if not generated or not expected_input_output_pairs:
        return 0.0
    n = min(len(generated), len(expected_input_output_pairs))
    passed = sum(1 for pair in generated[:n] if pair.get("passed", False))
    return passed / len(expected_input_output_pairs)


def measure_execution_time(execution_result: ExecutionResult) -> float | None:
    return execution_result["runtime_ms"]


def measure_code_length(code_snippet_from_llm_response: str) -> dict:
    lines = code_snippet_from_llm_response.splitlines()
    loc = sum(1 for line in lines if line.strip())
    return {"loc": loc, "char_count": len(code_snippet_from_llm_response)}
