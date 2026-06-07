"""Assemble and run a code_snippet_from_llm_response with its execution_engine.

Generation (GPU node) and execution (Apptainer container) are separate steps on Goethe.
This module handles the execution side: assembling the executable and running it via
the MultiPL-E toolchain container, then parsing the per-case OK/FAIL output.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import TypedDict

_APPTAINER_IMAGE_ENV_VAR = "EVALUATION_SIF"
_APPTAINER_IMAGE_DEFAULT = "evaluation.sif"

_CPP_COMPILE_FLAGS = ["-O2", "-std=c++17", "-o", "{binary}", "{source}"]
_JAVA_COMPILE_FLAGS = ["-d", "{outdir}", "{source}"]


class ExecutionResult(TypedDict):
    compiled: bool
    diagnostics: str
    input_output_pairs_from_llm_generated_code: list[dict]
    runtime_ms: float | None


def assemble_executable(
    execution_engine: str,
    code_snippet_from_llm_response: str,
    language: str,
) -> str:
    """Combine the model's code with the execution_engine into a compilable source string.

    The resulting string is the executable_code_snippet_from_llm_response.
    """
    if language == "python":
        return code_snippet_from_llm_response + "\n" + execution_engine
    elif language == "cpp":
        return (
            "#include <bits/stdc++.h>\n"
            "using namespace std;\n\n"
            + code_snippet_from_llm_response
            + "\n"
            + execution_engine
        )
    elif language == "java":
        return code_snippet_from_llm_response + "\n" + execution_engine
    else:
        raise ValueError(f"Unsupported language: {language!r}")


def run(
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
            return _run_python(
                executable_code_snippet_from_llm_response, tmp, timeout_s
            )
        elif language == "cpp":
            return _run_cpp(executable_code_snippet_from_llm_response, tmp, timeout_s)
        elif language == "java":
            return _run_java(executable_code_snippet_from_llm_response, tmp, timeout_s)
        else:
            raise ValueError(f"Unsupported language: {language!r}")


# ---- Per-language runners ----


def _run_python(source: str, tmp: Path, timeout_s: float) -> ExecutionResult:
    src_file = tmp / "solution.py"
    src_file.write_text(source)
    return _execute(["python3", str(src_file)], timeout_s)


def _run_cpp(source: str, tmp: Path, timeout_s: float) -> ExecutionResult:
    src_file = tmp / "solution.cpp"
    binary = tmp / "solution"
    src_file.write_text(source)

    compile_result = _compile(
        _in_container(["g++", "-O2", "-std=c++17", str(src_file), "-o", str(binary)])
    )
    if not compile_result["compiled"]:
        return compile_result

    return _execute(_in_container([str(binary)]), timeout_s)


def _run_java(source: str, tmp: Path, timeout_s: float) -> ExecutionResult:
    src_file = tmp / "Solution.java"
    src_file.write_text(source)

    compile_result = _compile(_in_container(["javac", "-d", str(tmp), str(src_file)]))
    if not compile_result["compiled"]:
        return compile_result

    return _execute(_in_container(["java", "-cp", str(tmp), "Main"]), timeout_s)


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


def _execute(cmd: list[str], timeout_s: float) -> ExecutionResult:
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
