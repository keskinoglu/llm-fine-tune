"""Tests for build_evaluation_dataset: engine generation, split reproduction."""

from __future__ import annotations

import polars as pl
import pytest

from llm_fine_tune.execution_harness import datatypes
from llm_fine_tune.dataset.build_evaluation_dataset import (
    _build_execution_engine,
    _filter_input_output_pairs_convertible_to_datatypes_in_all_supported_languages,
    _has_no_expected_output,
    _held_out_parallel_ids,
)
from llm_fine_tune.dataset.build_instruct_dataset import (
    DEFAULT_SPLIT_SEED as INSTRUCT_SPLIT_SEED,
    DEFAULT_TEST_FRAC as INSTRUCT_TEST_FRAC,
)
from llm_fine_tune.dataset import splits

_PLAIN_NODE_TYPES = {"parameters": [], "return_value": datatypes.PLAIN}


# ---------------------------------------------------------------------------
# _filter_input_output_pairs_convertible_to_datatypes_in_all_supported_languages
# ---------------------------------------------------------------------------


def test_filter_all_convertible():
    pairs = [{"input": [1, 2], "expected": 3}]
    convertible, unconvertible = (
        _filter_input_output_pairs_convertible_to_datatypes_in_all_supported_languages(
            pairs, _PLAIN_NODE_TYPES
        )
    )
    assert len(convertible) == 1
    assert len(unconvertible) == 0


def test_filter_some_unconvertible():
    pairs = [
        {"input": [1, 2], "expected": 3},
        {"input": [1, 2], "expected": {"nested": "dict"}},
    ]
    convertible, unconvertible = (
        _filter_input_output_pairs_convertible_to_datatypes_in_all_supported_languages(
            pairs, _PLAIN_NODE_TYPES
        )
    )
    assert len(convertible) == 1
    assert len(unconvertible) == 1


def test_filter_none_convertible():
    pairs = [{"input": [1, 2], "expected": {"nested": "dict"}}]
    convertible, unconvertible = (
        _filter_input_output_pairs_convertible_to_datatypes_in_all_supported_languages(
            pairs, _PLAIN_NODE_TYPES
        )
    )
    assert len(convertible) == 0
    assert len(unconvertible) == 1


def test_filter_node_params_are_convertible():
    node_types = {"parameters": [datatypes.LIST_NODE], "return_value": datatypes.PLAIN}
    pairs = [{"input": [[1, 2, 3]], "expected": 5}]
    convertible, unconvertible = (
        _filter_input_output_pairs_convertible_to_datatypes_in_all_supported_languages(
            pairs, node_types
        )
    )
    assert len(convertible) == 1
    assert len(unconvertible) == 0


# ---------------------------------------------------------------------------
# _has_no_expected_output
# ---------------------------------------------------------------------------


def test_has_no_expected_output_true():
    pairs = [{"input": [1], "expected": None}, {"input": [2], "expected": None}]
    assert _has_no_expected_output(pairs, _PLAIN_NODE_TYPES) is True


def test_has_no_expected_output_false_when_has_values():
    pairs = [{"input": [1], "expected": 2}, {"input": [3], "expected": 4}]
    assert _has_no_expected_output(pairs, _PLAIN_NODE_TYPES) is False


def test_has_no_expected_output_false_for_node_return():
    pairs = [{"input": [[1, 2, 3]], "expected": [1, 2, 3]}]
    node_types = {
        "parameters": [datatypes.LIST_NODE],
        "return_value": datatypes.LIST_NODE,
    }
    assert _has_no_expected_output(pairs, node_types) is False


# ---------------------------------------------------------------------------
# _build_execution_engine
# ---------------------------------------------------------------------------


def _pairs():
    return [{"input": [3, 4], "expected": 7}]


def test_python_engine_contains_entry_point():
    engine = _build_execution_engine("python", _pairs(), _PLAIN_NODE_TYPES, "add")
    assert "add" in engine
    assert "_sol = Solution()" in engine


def test_cpp_engine_contains_entry_point():
    engine = _build_execution_engine("cpp", _pairs(), _PLAIN_NODE_TYPES, "add")
    assert "add" in engine
    assert "int main()" in engine


def test_java_engine_contains_entry_point():
    engine = _build_execution_engine("java", _pairs(), _PLAIN_NODE_TYPES, "add")
    assert "add" in engine
    assert "class Main" in engine


def test_engine_unsupported_language_raises():
    with pytest.raises(datatypes.UnsupportedInputOutputValue):
        _build_execution_engine("rust", _pairs(), _PLAIN_NODE_TYPES, "solve")


def test_engine_unsupported_value_raises():
    with pytest.raises(datatypes.UnsupportedInputOutputValue):
        _build_execution_engine(
            "cpp",
            [{"input": [{"nested": "dict"}], "expected": 0}],
            _PLAIN_NODE_TYPES,
            "solve",
        )


# ---------------------------------------------------------------------------
# Datatype conversion (in execution_harness/datatypes.py)
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
    assert datatypes.cpp_type(value) == expected_type


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
    assert datatypes.cpp_literal(value) == expected_literal


def test_cpp_type_unsupported_raises():
    with pytest.raises(datatypes.UnsupportedInputOutputValue):
        datatypes.cpp_type({"key": "val"})


def test_cpp_literal_unsupported_raises():
    with pytest.raises(datatypes.UnsupportedInputOutputValue):
        datatypes.cpp_literal({"key": "val"})


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
    assert datatypes.java_type(value) == expected_type


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
    assert datatypes.java_literal(value) == expected_literal


# ---------------------------------------------------------------------------
# _held_out_parallel_ids: reproduces the instruct split exactly
# ---------------------------------------------------------------------------


def _make_base_frame(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "parallel_id": list(range(1, n + 1)),
            "cpp": ["code"] * n,
            "java": ["code"] * n,
            "python": ["code"] * n,
        }
    )


def test_held_out_ids_match_instruct_test_split():
    """_held_out_parallel_ids must return the same ids as splits.split_by_key at build time."""
    base = _make_base_frame(100)
    held_out = _held_out_parallel_ids(base)

    _, test_side = splits.split_by_key(
        base, "parallel_id", INSTRUCT_TEST_FRAC, INSTRUCT_SPLIT_SEED
    )
    expected_ids = set(test_side["parallel_id"].to_list())

    assert held_out == expected_ids


def test_held_out_fraction_is_approximately_correct():
    base = _make_base_frame(200)
    held_out = _held_out_parallel_ids(base)
    ratio = len(held_out) / 200
    assert 0.25 <= ratio <= 0.35, (
        f"held-out fraction {ratio:.2f} outside expected range"
    )


def test_held_out_excludes_single_language_snippets():
    """Snippets with only one language are ineligible for instruct and must be excluded."""
    base = pl.DataFrame(
        {
            "parallel_id": [1, 2, 3],
            "cpp": ["x", None, "x"],
            "java": ["x", None, None],
            "python": [None, "x", "x"],
        }
    )
    held_out = _held_out_parallel_ids(base)
    assert 2 not in held_out  # snippet 2 has only python
