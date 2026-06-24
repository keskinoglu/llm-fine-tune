"""Tests for compile_and_run_self_checking: trivially correct and wrong programs."""

from __future__ import annotations

import shutil

import pytest

from llm_fine_tune.execution_harness.execution import compile_and_run_self_checking


def _g_plus_plus_available() -> bool:
    return shutil.which("g++") is not None


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def test_python_passes():
    result = compile_and_run_self_checking("assert 1 + 1 == 2\n", "python")
    assert result["compiled"] is True
    assert result["passed"] is True
    assert result["returncode"] == 0


def test_python_fails():
    result = compile_and_run_self_checking("assert 1 + 1 == 3\n", "python")
    assert result["compiled"] is True
    assert result["passed"] is False
    assert result["returncode"] != 0


def test_python_syntax_error():
    # python3 runs the file; SyntaxError surfaces as a non-zero exit, not a compile_fail
    result = compile_and_run_self_checking("def broken(\n", "python")
    assert result["compiled"] is True
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# C++  (host must have g++)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _g_plus_plus_available(), reason="g++ not available on this host"
)
def test_cpp_passes():
    source = "#include <cassert>\nint main() { assert(1 + 1 == 2); return 0; }\n"
    result = compile_and_run_self_checking(source, "cpp")
    assert result["compiled"] is True
    assert result["passed"] is True


@pytest.mark.skipif(
    not _g_plus_plus_available(), reason="g++ not available on this host"
)
def test_cpp_fails():
    source = "#include <cassert>\nint main() { assert(1 + 1 == 3); return 0; }\n"
    result = compile_and_run_self_checking(source, "cpp")
    assert result["compiled"] is True
    assert result["passed"] is False


@pytest.mark.skipif(
    not _g_plus_plus_available(), reason="g++ not available on this host"
)
def test_cpp_compile_error():
    result = compile_and_run_self_checking(
        "int main() { this is not valid C++; }\n", "cpp"
    )
    assert result["compiled"] is False
    assert result["passed"] is False
    assert result["diagnostics"]


# ---------------------------------------------------------------------------
# Rust  (host must have rustc; runs in the docker/apptainer eval image)
# ---------------------------------------------------------------------------


def _rustc_available() -> bool:
    return shutil.which("rustc") is not None


@pytest.mark.skipif(not _rustc_available(), reason="rustc not available on this host")
def test_rust_passes():
    result = compile_and_run_self_checking(
        "fn main() { assert_eq!(1 + 1, 2); }\n", "rust"
    )
    assert result["compiled"] is True
    assert result["passed"] is True


@pytest.mark.skipif(not _rustc_available(), reason="rustc not available on this host")
def test_rust_fails():
    result = compile_and_run_self_checking(
        "fn main() { assert_eq!(1 + 1, 3); }\n", "rust"
    )
    assert result["compiled"] is True
    assert result["passed"] is False


@pytest.mark.skipif(not _rustc_available(), reason="rustc not available on this host")
def test_rust_compile_error():
    result = compile_and_run_self_checking("fn main() { not valid rust }\n", "rust")
    assert result["compiled"] is False
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# Go  (host must have go; uses `go test`, runs in the docker/apptainer eval image)
# ---------------------------------------------------------------------------


def _go_available() -> bool:
    return shutil.which("go") is not None


_GO_PASS = 'package m\n\nimport "testing"\n\nfunc TestX(t *testing.T) {\n\tif 1+1 != 2 {\n\t\tt.Error("no")\n\t}\n}\n'
_GO_FAIL = 'package m\n\nimport "testing"\n\nfunc TestX(t *testing.T) {\n\tif 1+1 != 3 {\n\t\tt.Error("no")\n\t}\n}\n'


@pytest.mark.skipif(not _go_available(), reason="go not available on this host")
def test_go_passes():
    result = compile_and_run_self_checking(_GO_PASS, "go")
    assert result["compiled"] is True
    assert result["passed"] is True


@pytest.mark.skipif(not _go_available(), reason="go not available on this host")
def test_go_fails():
    result = compile_and_run_self_checking(_GO_FAIL, "go")
    assert result["compiled"] is True
    assert result["passed"] is False
