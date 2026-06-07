"""Tests for build_evaluation_dataset: I/O parsing, engine generation, split reproduction."""

from __future__ import annotations

import json

import polars as pl
import pytest

from llm_fine_tune.dataset.build_evaluation_dataset import (
    UnparseableInputOutputPairs,
    UnsupportedInputOutputValue,
    _build_execution_engine,
    _cpp_literal,
    _cpp_type,
    _held_out_code_snippet_ids,
    _java_literal,
    _java_type,
    _parse_input_output_pairs,
)
from llm_fine_tune.dataset.build_instruct_dataset import (
    DEFAULT_SPLIT_SEED as INSTRUCT_SPLIT_SEED,
    DEFAULT_TEST_FRAC as INSTRUCT_TEST_FRAC,
)
from llm_fine_tune.dataset import splits


# ---------------------------------------------------------------------------
# _parse_input_output_pairs
# ---------------------------------------------------------------------------


def test_parse_json_format():
    raw = json.dumps({"input": [[1, 2], [3, 4]], "output": [3, 7]})
    pairs = _parse_input_output_pairs(raw)
    assert pairs == [{"input": [1, 2], "expected": 3}, {"input": [3, 4], "expected": 7}]


def test_parse_python_literal_format():
    raw = "{'input': [[1, 2]], 'output': [3]}"
    pairs = _parse_input_output_pairs(raw)
    assert pairs == [{"input": [1, 2], "expected": 3}]


def test_parse_wraps_non_list_input():
    raw = json.dumps({"input": "hello", "output": 5})
    pairs = _parse_input_output_pairs(raw)
    assert pairs == [{"input": ["hello"], "expected": 5}]


def test_parse_null_raises():
    with pytest.raises(UnparseableInputOutputPairs):
        _parse_input_output_pairs(None)


def test_parse_malformed_raises():
    with pytest.raises(UnparseableInputOutputPairs):
        _parse_input_output_pairs("{not valid json or python}")


def test_parse_length_mismatch_raises():
    raw = json.dumps({"input": [[1], [2]], "output": [10]})
    with pytest.raises(UnparseableInputOutputPairs):
        _parse_input_output_pairs(raw)


def test_parse_missing_keys_raises():
    with pytest.raises(UnparseableInputOutputPairs):
        _parse_input_output_pairs(json.dumps({"x": 1}))


def test_parse_inputs_plural_form():
    raw = json.dumps({"inputs": [[5]], "outputs": [25]})
    pairs = _parse_input_output_pairs(raw)
    assert pairs == [{"input": [5], "expected": 25}]


# ---------------------------------------------------------------------------
# C++ type and literal generators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (True, "bool"),
        (42, "int"),
        (3.14, "double"),
        ("hello", "std::string"),
        ([1, 2, 3], "std::vector<int>"),
        (["a", "b"], "std::vector<std::string>"),
    ],
)
def test_cpp_type(value, expected_type):
    assert _cpp_type(value) == expected_type


@pytest.mark.parametrize(
    "value, expected_literal",
    [
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (-7, "-7"),
        (3.14, repr(3.14)),
        ("hello", '"hello"'),
        ([1, 2], "{1, 2}"),
        ([], "{}"),
    ],
)
def test_cpp_literal(value, expected_literal):
    assert _cpp_literal(value) == expected_literal


def test_cpp_type_unsupported_raises():
    with pytest.raises(UnsupportedInputOutputValue):
        _cpp_type({"key": "val"})


def test_cpp_literal_unsupported_raises():
    with pytest.raises(UnsupportedInputOutputValue):
        _cpp_literal({"key": "val"})


# ---------------------------------------------------------------------------
# Java type and literal generators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (True, "boolean"),
        (1, "int"),
        (1.5, "double"),
        ("hi", "String"),
        ([1, 2], "int[]"),
        ([True, False], "boolean[]"),
        (["x"], "String[]"),
    ],
)
def test_java_type(value, expected_type):
    assert _java_type(value) == expected_type


@pytest.mark.parametrize(
    "value, expected_literal",
    [
        (True, "true"),
        (42, "42"),
        (2.5, repr(2.5)),
        ("world", '"world"'),
        ([1, 2, 3], "new int[]{1, 2, 3}"),
        ([], "new int[]{}"),
    ],
)
def test_java_literal(value, expected_literal):
    assert _java_literal(value) == expected_literal


# ---------------------------------------------------------------------------
# _build_execution_engine
# ---------------------------------------------------------------------------


def _stub_snippet(entry_point: str = "solve") -> dict:
    return {
        "code_snippet_id": 1,
        "entry_point": entry_point,
        "cpp": "x",
        "java": "x",
        "python": "x",
    }


def _pairs():
    return [{"input": [3, 4], "expected": 7}]


def test_python_engine_contains_entry_point():
    engine = _build_execution_engine(_stub_snippet("add"), "python", _pairs())
    assert "add" in engine
    assert "_CASES" in engine


def test_cpp_engine_contains_entry_point():
    engine = _build_execution_engine(_stub_snippet("add"), "cpp", _pairs())
    assert "add" in engine
    assert "int main()" in engine


def test_java_engine_contains_entry_point():
    engine = _build_execution_engine(_stub_snippet("add"), "java", _pairs())
    assert "add" in engine
    assert "class Main" in engine


def test_engine_unsupported_language_raises():
    with pytest.raises(UnsupportedInputOutputValue):
        _build_execution_engine(_stub_snippet(), "rust", _pairs())


def test_engine_unsupported_value_raises():
    with pytest.raises(UnsupportedInputOutputValue):
        _build_execution_engine(
            _stub_snippet(), "cpp", [{"input": [{"nested": "dict"}], "expected": 0}]
        )


# ---------------------------------------------------------------------------
# _held_out_code_snippet_ids: reproduces the instruct split exactly
# ---------------------------------------------------------------------------


def _make_base_frame(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code_snippet_id": list(range(1, n + 1)),
            "cpp": ["code"] * n,
            "java": ["code"] * n,
            "python": ["code"] * n,
        }
    )


def test_held_out_ids_match_instruct_test_split():
    """_held_out_code_snippet_ids must return the same ids as splits.split_by_key at build time."""
    base = _make_base_frame(100)
    held_out = _held_out_code_snippet_ids(base)

    _, test_side = splits.split_by_key(
        base, "code_snippet_id", INSTRUCT_TEST_FRAC, INSTRUCT_SPLIT_SEED
    )
    expected_ids = set(test_side["code_snippet_id"].to_list())

    assert held_out == expected_ids


def test_held_out_fraction_is_approximately_correct():
    base = _make_base_frame(200)
    held_out = _held_out_code_snippet_ids(base)
    ratio = len(held_out) / 200
    assert 0.25 <= ratio <= 0.35, (
        f"held-out fraction {ratio:.2f} outside expected range"
    )


def test_held_out_excludes_single_language_snippets():
    """Snippets with only one language are ineligible for instruct and must be excluded."""
    base = pl.DataFrame(
        {
            "code_snippet_id": [1, 2, 3],
            "cpp": ["x", None, "x"],
            "java": ["x", None, None],
            "python": [None, "x", "x"],
        }
    )
    held_out = _held_out_code_snippet_ids(base)
    assert 2 not in held_out  # snippet 2 has only python
