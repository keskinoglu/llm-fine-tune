"""Assemble a code_snippet with its execution_engine, then compile and run it.

Generation (GPU node) and execution (Apptainer container) are separate steps on Goethe.
This module handles the execution side: assembling the code_snippet_with_execution_wiring
and running it via the MultiPL-E toolchain container, then parsing the per-case OK/FAIL
output. The code_snippet is provenance-agnostic — the model's translation in eval, the
reference in dataset validation.
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
    input_output_pairs_from_code_snippet: list[dict]
    runtime_ms: float | None


def language_preamble(language: str) -> str:
    """Return the language-specific import preamble (no node definitions) for *language*."""
    if language == "cpp":
        return "#include <bits/stdc++.h>\nusing namespace std;"
    if language == "python":
        # LeetCode solutions assume these are pre-imported (no explicit imports in submissions).
        return (
            "import bisect, collections, functools, heapq, itertools, math, string\n"
            "from bisect import bisect_left, bisect_right, insort_left, insort_right\n"
            "from collections import Counter, defaultdict, deque, OrderedDict\n"
            "from functools import reduce, lru_cache, cache\n"
            "from heapq import heappush, heappop, heapify, nlargest, nsmallest\n"
            "from itertools import product, permutations, combinations, combinations_with_replacement, accumulate\n"
            "from math import gcd, sqrt, log, floor, ceil, inf, factorial, isqrt\n"
            "from typing import Dict, List, Optional, Set, Tuple"
        )
    if language == "java":
        # walkccc/LeetCode Java assumes java.util.*, java.util.stream.*, and
        # java.util.function.* are in scope (LeetCode auto-imports them).
        return (
            "import java.util.*;\n"
            "import java.util.stream.*;\n"
            "import java.util.function.*;"
        )
    raise ValueError(f"Unsupported language: {language!r}")


def assemble_code_snippet_with_execution_wiring(
    code_snippet: str,
    execution_engine: str,
    language: str,
) -> str:
    """Combine node definitions + the code_snippet + execution_engine into a compilable source string.

    The resulting string is the code_snippet_with_execution_wiring. Prepends
    ListNode/TreeNode definitions when needed, and language boilerplate.
    """
    definitions = datatypes.node_definitions(language)
    body = code_snippet + "\n" + execution_engine
    if language == "cpp":
        return language_preamble(language) + "\n\n" + definitions + "\n" + body
    if language == "python":
        return language_preamble(language) + "\n\n" + definitions + "\n" + body
    if language == "java":
        pair = "" if _JAVA_DEFINES_PAIR.search(body) else _JAVA_PAIR_DEF
        return language_preamble(language) + "\n\n" + pair + definitions + "\n" + body
    return definitions + "\n" + body


def compile_and_run(
    code_snippet_with_execution_wiring: str,
    language: str,
    *,
    timeout_s: float = 10.0,
) -> ExecutionResult:
    """Compile and run the code_snippet_with_execution_wiring; return an ExecutionResult.

    Runs inside the Apptainer image if EVALUATION_SIF is set; falls back to bare host for
    Python (unit-test convenience). C++/Java always require the container on Goethe.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if language == "python":
            return _execute_python(code_snippet_with_execution_wiring, tmp, timeout_s)
        elif language == "cpp":
            return _execute_cpp(code_snippet_with_execution_wiring, tmp, timeout_s)
        elif language == "java":
            return _execute_java(code_snippet_with_execution_wiring, tmp, timeout_s)
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
        "input_output_pairs_from_code_snippet": [],
        "runtime_ms": None,
    }


def _run_subprocess(cmd: list[str], timeout_s: float) -> ExecutionResult:
    start = time.perf_counter()
    try:
        # errors="replace": untrusted model code can emit non-UTF-8 bytes on stdout/stderr;
        # decode lossily so one such row scores as wrong_output instead of crashing the run.
        proc = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return _failed("Execution timed out")
    runtime_ms = (time.perf_counter() - start) * 1000

    if proc.returncode not in (0, 1):
        return _failed(proc.stderr or f"exit code {proc.returncode}")

    pairs = _parse_execution_stdout(proc.stdout)
    return {
        "compiled": True,
        "diagnostics": proc.stderr,
        "input_output_pairs_from_code_snippet": pairs,
        "runtime_ms": runtime_ms,
    }


def _parse_execution_stdout(stdout: str) -> list[dict]:
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
        "input_output_pairs_from_code_snippet": [],
        "runtime_ms": None,
    }


# ---- Self-checking runner (MultiPL-E / pass@1 style) ----

_PUBLIC_CLASS_RE = re.compile(r"public\s+(?:final\s+)?class\s+(\w+)")
_ANY_CLASS_RE = re.compile(r"\bclass\s+(\w+)")

# Provisioned in the eval image (evaluation_image.def). MultiPL-E java imports org.javatuples.*.
_JAVATUPLES_JAR = "/opt/javatuples.jar"


def _detect_java_class(source: str) -> str:
    m = _PUBLIC_CLASS_RE.search(source)
    if m:
        return m.group(1)
    m = _ANY_CLASS_RE.search(source)
    if m:
        return m.group(1)
    return "Main"


def _run_raw(cmd: list[str], timeout_s: float) -> dict:
    """Run *cmd* and return raw exit-code result (not OK/FAIL parsing)."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return {
            "compiled": True,
            "passed": False,
            "returncode": -1,
            "diagnostics": "Execution timed out",
            "runtime_ms": None,
        }
    runtime_ms = (time.perf_counter() - start) * 1000
    passed = proc.returncode == 0
    return {
        "compiled": True,
        "passed": passed,
        "returncode": proc.returncode,
        "diagnostics": proc.stderr or proc.stdout,
        "runtime_ms": runtime_ms,
    }


def _self_check_failed(diagnostics: str) -> dict:
    return {
        "compiled": False,
        "passed": False,
        "returncode": -1,
        "diagnostics": diagnostics,
        "runtime_ms": None,
    }


def _self_check_python(source: str, tmp: Path, timeout_s: float) -> dict:
    src_file = tmp / "solution.py"
    src_file.write_text(source)
    return _run_raw(_in_container(["python3", str(src_file)]), timeout_s)


def _self_check_cpp(source: str, tmp: Path, timeout_s: float) -> dict:
    src_file = tmp / "solution.cpp"
    binary = tmp / "solution"
    src_file.write_text(source)
    compile_result = _compile(
        _in_container(["g++", "-O2", "-std=c++20", str(src_file), "-o", str(binary)])
    )
    if not compile_result["compiled"]:
        return _self_check_failed(compile_result["diagnostics"])
    return _run_raw(_in_container([str(binary)]), timeout_s)


def _self_check_java(source: str, tmp: Path, timeout_s: float) -> dict:
    class_name = _detect_java_class(source)
    src_file = tmp / f"{class_name}.java"
    src_file.write_text(source)
    # MultiPL-E java prompts unconditionally `import org.javatuples.*` (tuple return types);
    # the jar is provisioned in the eval image. Absent on the host, where java isn't run.
    classpath = f"{tmp}:{_JAVATUPLES_JAR}"
    compile_result = _compile(
        _in_container(["javac", "-cp", classpath, "-d", str(tmp), str(src_file)])
    )
    if not compile_result["compiled"]:
        return _self_check_failed(compile_result["diagnostics"])
    return _run_raw(
        _in_container(
            [
                "java",
                "-ea",  # MultiPL-E tests use `assert`; without -ea the JVM ignores them (false pass)
                "-Xmx512m",
                "-XX:+UseSerialGC",
                "-XX:ActiveProcessorCount=1",
                "-cp",
                classpath,
                class_name,
            ]
        ),
        timeout_s,
    )


def compile_and_run_self_checking(
    source: str,
    language: str,
    *,
    timeout_s: float = 15.0,
) -> dict:
    """Compile and run a self-checking program; pass = exit 0.

    Returns ``{"compiled": bool, "passed": bool, "returncode": int,
    "diagnostics": str, "runtime_ms": float | None}``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if language == "python":
            return _self_check_python(source, tmp, timeout_s)
        elif language == "cpp":
            return _self_check_cpp(source, tmp, timeout_s)
        elif language == "java":
            return _self_check_java(source, tmp, timeout_s)
        else:
            raise ValueError(f"Unsupported language: {language!r}")
