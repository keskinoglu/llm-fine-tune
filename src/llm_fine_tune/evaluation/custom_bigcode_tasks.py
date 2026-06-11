"""bigcode Task subclass for code_snippet_translation evaluation.

CodeSnippetTranslationTask drives the held-out evaluation dataset through bigcode's
generate → execute → score pipeline. Imported only by run_bigcode_cli.py.
"""

from __future__ import annotations

import json

from bigcode_eval.base import Task

from llm_fine_tune.execution_harness import execution
from llm_fine_tune.evaluation import score
from llm_fine_tune.evaluation import extract_code_snippet_from_llm_response as extractor


class CodeSnippetTranslationTask(Task):
    DATASET_PATH = "tkeskin/leetcode-solutions"
    DATASET_NAME = "evaluation"

    def __init__(self) -> None:
        super().__init__(stop_words=["```"], requires_execution=True)

    def get_dataset(self):
        return self.dataset["test"]

    def get_prompt(self, payload: dict) -> str:
        return f"{payload['user_prompt']}\n\n{payload['code_snippet_to_translate']}"

    def get_reference(self, payload: dict) -> str:
        return payload["expected_code_snippet_translation"]

    def postprocess_generation(self, llm_response: str, i: int) -> str:
        target_language = self.get_dataset()[i]["target_language"]
        return extractor.extract(llm_response, target_language)

    def process_results(
        self, generations: list[list[str]], references: list[str]
    ) -> dict:
        records = [
            _score_single_sample(
                self.get_dataset()[i],
                generation_list[0] if generation_list else "",
            )
            for i, generation_list in enumerate(generations)
        ]
        aggregates = _aggregate(records)
        aggregates["per_sample"] = records
        return aggregates


# ---- Helpers ----


def _score_single_sample(payload: dict, code_snippet_from_llm_response: str) -> dict:
    expected_input_output_pairs = json.loads(payload["expected_input_output_pairs"])
    executable = execution.build_executable_code_snippet_from_llm_response(
        payload["execution_engine"],
        code_snippet_from_llm_response,
        payload["target_language"],
    )
    execution_result = execution.execute(executable, payload["target_language"])
    sample_scores = score.score(
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


def _aggregate(records: list[dict]) -> dict:
    if not records:
        return {"pass@1": 0.0, "compiled": 0.0, "test_pass_rate": 0.0}
    n = len(records)
    return {
        "pass@1": sum(r.get("pass@1", 0.0) for r in records) / n,
        "compiled": sum(r.get("compiled", 0.0) for r in records) / n,
        "test_pass_rate": sum(r.get("test_pass_rate", 0.0) for r in records) / n,
    }
