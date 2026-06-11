"""Assemble and run a code_snippet_from_llm_response with its execution_engine.

Generation (GPU node) and execution (Apptainer container) are separate steps on Goethe.
This module handles the execution side: assembling the executable and running it via
the MultiPL-E toolchain container, then parsing the per-case OK/FAIL output.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TypedDict

from llm_fine_tune.execution_harness import datatypes

_APPTAINER_IMAGE_ENV_VAR = "EVALUATION_SIF"
_APPTAINER_IMAGE_DEFAULT = "evaluation.sif"

# LeetCode's Java environment provides javafx.util.Pair (gone from the JDK since 11).
# walkccc solutions use its getKey()/getValue() API; ~5% of Java references need it.
# Prepended only when the code doesn't define its own Pair (avoids a collision in the
# model-evaluation path).
_JAVA_PAIR_DEF = """
class Pair<K, V> {
    private final K key;
    private final V value;
    public Pair(K key, V value) { this.key = key; this.value = value; }
    public K getKey() { return key; }
    public V getValue() { return value; }
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Pair)) return false;
        Pair<?, ?> p = (Pair<?, ?>) o;
        return java.util.Objects.equals(key, p.key) && java.util.Objects.equals(value, p.value);
    }
    public int hashCode() { return java.util.Objects.hash(key, value); }
    public String toString() { return "(" + key + ", " + value + ")"; }
}
"""

_JAVA_DEFINES_PAIR = re.compile(r"\b(?:class|record|interface)\s+Pair\b")


class ExecutionResult(TypedDict):
    compiled: bool
    diagnostics: str
    input_output_pairs_from_llm_generated_code: list[dict]
    runtime_ms: float | None


def build_executable_code_snippet_from_llm_response(
    execution_engine: str,
    code_snippet_from_llm_response: str,
    target_language: str,
) -> str:
    """Combine node definitions + model's code + execution_engine into a compilable source string.

    The resulting string is the executable_code_snippet_from_llm_response.
    Prepends ListNode/TreeNode definitions when needed, and language boilerplate.
    """
    definitions = datatypes.node_definitions(target_language)
    body = code_snippet_from_llm_response + "\n" + execution_engine
    if target_language == "cpp":
        return (
            "#include <bits/stdc++.h>\nusing namespace std;\n\n"
            + definitions
            + "\n"
            + body
        )
    if target_language == "python":
        # LeetCode solutions assume these are pre-imported (no explicit imports in submissions).
        stdlib = (
            "import bisect, collections, functools, heapq, itertools, math, string\n"
            "from bisect import bisect_left, bisect_right, insort_left, insort_right\n"
            "from collections import Counter, defaultdict, deque, OrderedDict\n"
            "from functools import reduce, lru_cache, cache\n"
            "from heapq import heappush, heappop, heapify, nlargest, nsmallest\n"
            "from itertools import product, permutations, combinations, combinations_with_replacement, accumulate\n"
            "from math import gcd, sqrt, log, floor, ceil, inf, factorial, isqrt\n"
            "from typing import Dict, List, Optional, Set, Tuple\n"
        )
        return stdlib + "\n" + definitions + "\n" + body
    if target_language == "java":
        # walkccc/LeetCode Java assumes java.util.*, java.util.stream.*, and
        # java.util.function.* are in scope (LeetCode auto-imports them).
        imports = (
            "import java.util.*;\n"
            "import java.util.stream.*;\n"
            "import java.util.function.*;\n\n"
        )
        pair = "" if _JAVA_DEFINES_PAIR.search(body) else _JAVA_PAIR_DEF
        return imports + pair + definitions + "\n" + body
    return definitions + "\n" + body


def execute(
    executable_code_snippet_from_llm_response: str,
    language: str,
    *,
    timeout_s: float = 10.0,
) -> ExecutionResult:
    """Compile and run the executable_code_snippet_from_llm_response; return an ExecutionResult.

    Runs inside the Apptainer image if EVALUATION_SIF is set; falls back to bare host for
    Python (unit-test convenience). C++/Java always require the container on Goethe.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if language == "python":
            return _execute_python(
                executable_code_snippet_from_llm_response, tmp, timeout_s
            )
        elif language == "cpp":
            return _execute_cpp(
                executable_code_snippet_from_llm_response, tmp, timeout_s
            )
        elif language == "java":
            return _execute_java(
                executable_code_snippet_from_llm_response, tmp, timeout_s
            )
        else:
            raise ValueError(f"Unsupported language: {language!r}")


# ---- Per-language runners ----


def _execute_python(source: str, tmp: Path, timeout_s: float) -> ExecutionResult:
    src_file = tmp / "solution.py"
    src_file.write_text(source)
    return _run_subprocess(["python3", str(src_file)], timeout_s)


def _execute_cpp(source: str, tmp: Path, timeout_s: float) -> ExecutionResult:
    src_file = tmp / "solution.cpp"
    binary = tmp / "solution"
    src_file.write_text(source)

    compile_result = _compile(
        _in_container(["g++", "-O2", "-std=c++20", str(src_file), "-o", str(binary)])
    )
    if not compile_result["compiled"]:
        return compile_result

    return _run_subprocess(_in_container([str(binary)]), timeout_s)


def _execute_java(source: str, tmp: Path, timeout_s: float) -> ExecutionResult:
    src_file = tmp / "Solution.java"
    src_file.write_text(source)

    compile_result = _compile(_in_container(["javac", "-d", str(tmp), str(src_file)]))
    if not compile_result["compiled"]:
        return compile_result

    return _run_subprocess(
        _in_container(
            [
                "java",
                "-Xmx512m",
                "-XX:+UseSerialGC",
                "-XX:ActiveProcessorCount=1",
                "-cp",
                str(tmp),
                "Main",
            ]
        ),
        timeout_s,
    )


# ---- Subprocess helpers ----


def _in_container(cmd: list[str]) -> list[str]:
    """Wrap cmd in apptainer exec if EVALUATION_SIF is set."""
    import os

    sif = os.environ.get(_APPTAINER_IMAGE_ENV_VAR, "")
    if sif:
        return ["apptainer", "exec", "--net", "none", sif] + cmd
    return cmd


def _compile(cmd: list[str]) -> ExecutionResult:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return _failed("Compilation timed out")
    if proc.returncode != 0:
        return _failed(proc.stderr or proc.stdout)
    return {
        "compiled": True,
        "diagnostics": "",
        "input_output_pairs_from_llm_generated_code": [],
        "runtime_ms": None,
    }


def _run_subprocess(cmd: list[str], timeout_s: float) -> ExecutionResult:
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return _failed("Execution timed out")
    runtime_ms = (time.perf_counter() - start) * 1000

    if proc.returncode not in (0, 1):
        return _failed(proc.stderr or f"exit code {proc.returncode}")

    pairs = _parse_output(proc.stdout)
    return {
        "compiled": True,
        "diagnostics": proc.stderr,
        "input_output_pairs_from_llm_generated_code": pairs,
        "runtime_ms": runtime_ms,
    }


def _parse_output(stdout: str) -> list[dict]:
    """Parse OK/FAIL lines from the execution_engine's stdout."""
    pairs = []
    for line in stdout.splitlines():
        line = line.strip()
        if line == "OK":
            pairs.append({"passed": True})
        elif line.startswith("FAIL"):
            pairs.append({"passed": False})
    return pairs


def _failed(diagnostics: str) -> ExecutionResult:
    return {
        "compiled": False,
        "diagnostics": diagnostics,
        "input_output_pairs_from_llm_generated_code": [],
        "runtime_ms": None,
    }
