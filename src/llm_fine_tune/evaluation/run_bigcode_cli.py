"""Register CodeSnippetTranslationTask with bigcode's TASK_REGISTRY, then hand off to bigcode's CLI."""

from __future__ import annotations


def main() -> None:
    from bigcode_eval.tasks import TASK_REGISTRY

    from llm_fine_tune.evaluation.custom_bigcode_tasks import CodeSnippetTranslationTask

    TASK_REGISTRY["code_snippet_translation"] = CodeSnippetTranslationTask

    from bigcode_eval.main import main as _bigcode_main

    _bigcode_main()


if __name__ == "__main__":
    main()
