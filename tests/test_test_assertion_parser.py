"""Tests for dataset/quality/test_assertion_parser: typed (input, expected) extraction."""

from __future__ import annotations


from llm_fine_tune.dataset.quality.test_assertion_parser import parse_test_cases


def _check(source: str) -> list[dict]:
    return parse_test_cases(f"def check(candidate):\n    {source}")


# ---------------------------------------------------------------------------
# Plain compare (candidate(...) == expected)
# ---------------------------------------------------------------------------


def test_plain_int_expected():
    cases = _check("assert candidate(3, 4) == 7")
    assert cases == [{"input": [3, 4], "expected": 7}]


def test_plain_string_expected_preserves_type():
    cases = _check('assert candidate(231132) == "231132"')
    assert cases == [{"input": [231132], "expected": "231132"}]


def test_plain_bool_expected():
    cases = _check("assert candidate(5) == True")
    assert cases == [{"input": [5], "expected": True}]


def test_plain_list_expected():
    cases = _check("assert candidate([1, 2]) == [1, 2, 3]")
    assert cases == [{"input": [[1, 2]], "expected": [1, 2, 3]}]


def test_multiple_assert_lines():
    src = "\n    ".join(
        [
            "assert candidate(1) == 1",
            "assert candidate(2) == 4",
            "assert candidate(3) == 9",
        ]
    )
    cases = _check(src)
    assert len(cases) == 3
    assert cases[1] == {"input": [2], "expected": 4}


def test_kwargs_in_signature():
    cases = _check("assert candidate(nums=[1,2,3], target=9) == 0")
    assert cases == [{"input": [[1, 2, 3], 9], "expected": 0}]


# ---------------------------------------------------------------------------
# list_node / tree_node unwrapping
# ---------------------------------------------------------------------------


def test_is_same_list_unwraps_node():
    cases = _check("assert is_same_list(candidate([1, 2]), list_node([1, 2, 3]))")
    assert cases == [{"input": [[1, 2]], "expected": [1, 2, 3]}]


def test_is_same_tree_unwraps_node():
    cases = _check(
        "assert is_same_tree(candidate([1, None, 2]), tree_node([1, None, 2, 3]))"
    )
    assert cases == [{"input": [[1, None, 2]], "expected": [1, None, 2, 3]}]


def test_list_node_input_is_unwrapped():
    cases = _check("assert candidate(list_node([1, 2, 3])) == 6")
    assert cases == [{"input": [[1, 2, 3]], "expected": 6}]


# ---------------------------------------------------------------------------
# Real-shape coverage from the source dataset
# ---------------------------------------------------------------------------


_REAL_TEST = """
def check(candidate):
    assert candidate(3, 4) == 7
    assert candidate(-1, 1) == 0
    assert candidate(0, 0) == 0
""".strip()


def test_real_shape_three_cases():
    cases = parse_test_cases(_REAL_TEST)
    assert len(cases) == 3
    assert cases[0] == {"input": [3, 4], "expected": 7}
    assert cases[2] == {"input": [0, 0], "expected": 0}


def test_empty_test_source():
    assert parse_test_cases("") == []


def test_no_asserts():
    assert parse_test_cases("def check(candidate):\n    pass") == []
