"""Tests for supported_language_datatypes: literals, types, node definitions, node-type detection."""

from __future__ import annotations

import pytest

from llm_fine_tune.execution_harness import datatypes


# ---------------------------------------------------------------------------
# Plain-type literal/type functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "true"),
        (False, "false"),
        (42, "42"),
        (-7, "-7"),
        (3.14, repr(3.14)),
        ("hello", '"hello"'),
        ([1, 2, 3], "{1, 2, 3}"),
        ([], "{}"),
        (["a", "b"], '{"a", "b"}'),
    ],
)
def test_cpp_literal(value, expected):
    assert datatypes.cpp_literal(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "bool"),
        (42, "int"),
        (3.14, "double"),
        ("hello", "std::string"),
        ([1, 2, 3], "std::vector<int>"),
        (["a", "b"], "std::vector<std::string>"),
        ([], "std::vector<int>"),
    ],
)
def test_cpp_type(value, expected):
    assert datatypes.cpp_type(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "true"),
        (42, "42"),
        (3.14, repr(3.14)),
        ("hello", '"hello"'),
        ([1, 2, 3], "new int[]{1, 2, 3}"),
        ([], "new int[]{}"),
        (["a", "b"], 'new String[]{"a", "b"}'),
    ],
)
def test_java_literal(value, expected):
    assert datatypes.java_literal(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "boolean"),
        (42, "int"),
        (3.14, "double"),
        ("hello", "String"),
        ([1, 2, 3], "int[]"),
        ([True, False], "boolean[]"),
        (["a", "b"], "String[]"),
        ([], "int[]"),
    ],
)
def test_java_type(value, expected):
    assert datatypes.java_type(value) == expected


def test_python_literal():
    assert datatypes.python_literal(42) == "42"
    assert datatypes.python_literal("hello") == "'hello'"
    assert datatypes.python_literal([1, 2, 3]) == "[1, 2, 3]"
    assert datatypes.python_literal(True) == "True"


def test_unsupported_cpp_literal_raises():
    with pytest.raises(datatypes.UnsupportedInputOutputValue):
        datatypes.cpp_literal({"key": "val"})


def test_unsupported_cpp_type_raises():
    with pytest.raises(datatypes.UnsupportedInputOutputValue):
        datatypes.cpp_type({"key": "val"})


def test_unsupported_java_literal_raises():
    with pytest.raises(datatypes.UnsupportedInputOutputValue):
        datatypes.java_literal({"key": "val"})


# ---------------------------------------------------------------------------
# node_definitions per language
# ---------------------------------------------------------------------------


def test_python_node_defs_contains_listnode():
    defs = datatypes.node_definitions("python")
    assert "class ListNode" in defs
    assert "class TreeNode" in defs
    assert "from_array" in defs
    assert "to_array" in defs


def test_cpp_node_defs_contains_listnode():
    defs = datatypes.node_definitions("cpp")
    assert "struct ListNode" in defs
    assert "struct TreeNode" in defs


def test_java_node_defs_contains_listnode():
    defs = datatypes.node_definitions("java")
    assert "class ListNode" in defs
    assert "class TreeNode" in defs


def test_node_defs_unsupported_language_raises():
    with pytest.raises(datatypes.UnsupportedInputOutputValue):
        datatypes.node_definitions("rust")


# ---------------------------------------------------------------------------
# list_node_array_text / tree_node_array_text
# ---------------------------------------------------------------------------


def test_list_node_array_text():
    assert datatypes.list_node_array_text([1, 2, 3]) == "[1, 2, 3]"


def test_tree_node_array_text_trims_trailing_nulls():
    assert datatypes.tree_node_array_text([1, None, 2, None, None]) == "[1, None, 2]"


def test_tree_node_array_text_no_trailing_nulls():
    assert datatypes.tree_node_array_text([1, None, 2]) == "[1, None, 2]"


def test_tree_node_array_text_all_none():
    assert datatypes.tree_node_array_text([None, None]) == "[]"


def test_tree_node_array_text_empty():
    assert datatypes.tree_node_array_text([]) == "[]"


# ---------------------------------------------------------------------------
# detect_node_types
# ---------------------------------------------------------------------------


def test_detect_plain_params():
    code_snippet = {
        "entry_point": "add",
        "python": "class Solution:\n    def add(self, a: int, b: int) -> int: return a + b",
    }
    types = datatypes.detect_node_types(code_snippet)
    assert types["parameters"] == [datatypes.PLAIN, datatypes.PLAIN]
    assert types["return_value"] == datatypes.PLAIN


def test_detect_list_node_param():
    code_snippet = {
        "entry_point": "reverseList",
        "python": "class Solution:\n    def reverseList(self, head: ListNode) -> ListNode: return head",
    }
    types = datatypes.detect_node_types(code_snippet)
    assert types["parameters"] == [datatypes.LIST_NODE]
    assert types["return_value"] == datatypes.LIST_NODE


def test_detect_tree_node_param():
    code_snippet = {
        "entry_point": "invertTree",
        "python": "class Solution:\n    def invertTree(self, root: TreeNode) -> TreeNode: return root",
    }
    types = datatypes.detect_node_types(code_snippet)
    assert types["parameters"] == [datatypes.TREE_NODE]
    assert types["return_value"] == datatypes.TREE_NODE


def test_detect_optional_list_node():
    code_snippet = {
        "entry_point": "detectCycle",
        "python": "class Solution:\n    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]: return head",
    }
    types = datatypes.detect_node_types(code_snippet)
    assert types["parameters"] == [datatypes.LIST_NODE]
    assert types["return_value"] == datatypes.LIST_NODE


def test_detect_union_none():
    code_snippet = {
        "entry_point": "detectCycle",
        "python": "class Solution:\n    def detectCycle(self, head: ListNode | None) -> ListNode | None: return head",
    }
    types = datatypes.detect_node_types(code_snippet)
    assert types["parameters"] == [datatypes.LIST_NODE]
    assert types["return_value"] == datatypes.LIST_NODE


def test_detect_no_python_solution():
    code_snippet = {
        "entry_point": "solve",
        "python": None,
    }
    with pytest.raises(
        datatypes.UnsupportedInputOutputValue, match="No Python reference solution"
    ):
        datatypes.detect_node_types(code_snippet)


def test_detect_no_entry_point_found():
    code_snippet = {
        "entry_point": "nonexistent",
        "python": "class Solution:\n    def solve(self, x: int) -> int: return x",
    }
    # Falls back to all PLAIN
    types = datatypes.detect_node_types(code_snippet)
    assert types["parameters"] == []
    assert types["return_value"] == datatypes.PLAIN


# ---------------------------------------------------------------------------
# ListNode.from_array / to_array round-trip (integration)
# ---------------------------------------------------------------------------


def test_list_node_python_roundtrip():
    exec(datatypes.node_definitions("python"), globals_copy := {})
    ListNode = globals_copy["ListNode"]
    node = ListNode.from_array([1, 2, 3])
    assert node.to_array() == [1, 2, 3]
    assert ListNode.from_array([]) is None
    node = ListNode.from_array([42])
    assert node.to_array() == [42]


# ---------------------------------------------------------------------------
# TreeNode.from_array / to_array round-trip (integration)
# ---------------------------------------------------------------------------


def test_tree_node_python_roundtrip():
    exec(datatypes.node_definitions("python"), globals_copy := {})
    TreeNode = globals_copy["TreeNode"]
    node = TreeNode.from_array([1, None, 2])
    assert node.to_array() == [1, None, 2]
    assert TreeNode.from_array([]) is None
    node = TreeNode.from_array([42])
    assert node.to_array() == [42]


# ---------------------------------------------------------------------------
# cpp_optional_int_vector_literal / java_nullable_int_array_literal
# ---------------------------------------------------------------------------


def test_cpp_optional_int_vector_literal_plain():
    assert datatypes.cpp_optional_int_vector_literal([1, 2, 3]) == "{1, 2, 3}"


def test_cpp_optional_int_vector_literal_with_nullopt():
    assert (
        datatypes.cpp_optional_int_vector_literal([1, None, 2])
        == "{1, std::nullopt, 2}"
    )


def test_cpp_optional_int_vector_literal_trims_trailing_none():
    assert datatypes.cpp_optional_int_vector_literal([1, 2, None, None]) == "{1, 2}"


def test_java_nullable_int_array_literal_plain():
    assert (
        datatypes.java_nullable_int_array_literal([1, 2, 3]) == "new Integer[]{1, 2, 3}"
    )


def test_java_nullable_int_array_literal_with_null():
    assert (
        datatypes.java_nullable_int_array_literal([1, None, 2])
        == "new Integer[]{1, null, 2}"
    )


def test_java_nullable_int_array_literal_trims_trailing_none():
    assert (
        datatypes.java_nullable_int_array_literal([1, 2, None]) == "new Integer[]{1, 2}"
    )


# ---------------------------------------------------------------------------
# Engine syntax validation (Python)
# ---------------------------------------------------------------------------


def test_python_engine_syntax_plain():
    from llm_fine_tune.dataset.build_evaluation_dataset import _build_execution_engine

    node_types = {
        "parameters": [datatypes.PLAIN, datatypes.PLAIN],
        "return_value": datatypes.PLAIN,
    }
    pairs = [{"input": [2, 7], "expected": 9}]
    engine = _build_execution_engine("python", pairs, node_types, "add")
    compile(engine, "<engine>", "exec")  # raises SyntaxError if broken


def test_python_engine_syntax_list_node():
    from llm_fine_tune.dataset.build_evaluation_dataset import _build_execution_engine

    node_types = {
        "parameters": [datatypes.LIST_NODE],
        "return_value": datatypes.LIST_NODE,
    }
    pairs = [{"input": [[1, 2, 3]], "expected": [3, 2, 1]}]
    engine = _build_execution_engine("python", pairs, node_types, "reverseList")
    compile(engine, "<engine>", "exec")


def test_python_engine_syntax_tree_node():
    from llm_fine_tune.dataset.build_evaluation_dataset import _build_execution_engine

    node_types = {
        "parameters": [datatypes.TREE_NODE],
        "return_value": datatypes.TREE_NODE,
    }
    pairs = [{"input": [[1, None, 2]], "expected": [2, None, 1]}]
    engine = _build_execution_engine("python", pairs, node_types, "mirrorTree")
    compile(engine, "<engine>", "exec")
