"""Strip prose and markdown fences from an llm_response to get code_snippet_from_llm_response."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```[A-Za-z+#]*\n(.*?)```", re.DOTALL)


def extract(llm_response: str, language: str) -> str:
    """Return the code_snippet_from_llm_response extracted from llm_response.

    Tries fenced blocks first (``` ... ```); falls back to the full response stripped.
    """
    match = _FENCE.search(llm_response)
    if match:
        return match.group(1).strip()
    return llm_response.strip()
