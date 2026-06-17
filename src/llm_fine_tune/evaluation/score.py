"""Compose all metric measures for one code_snippet_translation sample.

No bigcode dependency — shared by the bigcode task (eval driver) and the standalone
Phase-2 scorer (run_execution_scoring) that runs inside the --net --network none container.
"""

from __future__ import annotations

import json

from llm_fine_tune.evaluation import metrics
from llm_fine_tune.execution_harness import execution
from llm_fine_tune.execution_harness.execution import ExecutionResult


def score(
    code_snippet_from_llm_response: str,
    execution_result: ExecutionResult,
    expected_input_output_pairs: list[dict],
) -> dict[str, object]:
    """Run all metric measures and return a flat dict of scores for one sample."""
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


def score_bigcode_task_payload(
    payload: dict, code_snippet_from_llm_response: str
) -> dict:
    """Assemble + run + score one bigcode_task_payload against its expected pairs.

    Returns a per-sample record (identity columns + the score dict). This is the unit
    both the bigcode task's process_results and the standalone scorer produce.
    """
    expected_input_output_pairs = json.loads(payload["expected_input_output_pairs"])
    code_snippet_with_execution_wiring = (
        execution.assemble_code_snippet_with_execution_wiring(
            code_snippet_from_llm_response,
            payload["execution_engine"],
            payload["target_language"],
        )
    )
    execution_result = execution.compile_and_run(
        code_snippet_with_execution_wiring, payload["target_language"]
    )
    sample_scores = score(
        code_snippet_from_llm_response,
        execution_result,
        expected_input_output_pairs,
    )
    return {
        "parallel_id": payload["parallel_id"],
        "source_language": payload["source_language"],
        "target_language": payload["target_language"],
        "difficulty": payload.get("difficulty"),
        **sample_scores,
    }
