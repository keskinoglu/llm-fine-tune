"""Compose all metric measures for one code_snippet_translation sample."""

from __future__ import annotations


from llm_fine_tune.evaluation import metrics
from llm_fine_tune.evaluation.execution import ExecutionResult


def score(
    code_snippet_from_llm_response: str,
    execution_result: ExecutionResult,
    expected_input_output_pairs: list[dict],
) -> dict[str, object]:
    """Run all metric measures and return a flat dict suitable for bigcode's process_results."""
    code_length = metrics.measure_code_length(code_snippet_from_llm_response)
    correctness = metrics.measure_correctness(
        execution_result, expected_input_output_pairs
    )
    return {
        "compiled": metrics.measure_compilation_success(execution_result),
        "test_pass_rate": correctness,
        "pass@1": 1.0 if correctness == 1.0 else 0.0,
        "runtime_ms": metrics.measure_execution_time(execution_result),
        "loc": code_length["loc"],
        "char_count": code_length["char_count"],
    }
