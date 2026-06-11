"""Parse the newfacade `test` column into typed (input, expected) pairs.

The column is a `check(candidate)` function whose asserts take exactly three shapes
(verified across all 2,869 rows / 280,006 asserts, zero parse failures):

    assert candidate(<kwargs>) == <expected>                     # plain return value
    assert is_same_list(candidate(<kwargs>), list_node([...]))   # ListNode return
    assert is_same_tree(candidate(<kwargs>), tree_node([...]))   # TreeNode return

Unlike the `input_output` column, expected values keep their real Python type — no
inference required. The {"input": [values...], "expected": value} output shape is
exactly what the execution-engine builders in build_evaluation_dataset.py already consume.
"""

from __future__ import annotations

import ast

_NODE_WRAPPERS = {"list_node", "tree_node"}

# Bare names that appear in test assertions but are not Python keywords/literals.
_KNOWN_NAMES: dict[str, object] = {"inf": float("inf"), "nan": float("nan")}


def parse_test_cases(test_source: str) -> list[dict]:
    if not test_source:
        return []
    tree = ast.parse(test_source)
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            try:
                results.append(_parse_assert(node.test))
            except (ValueError, IndexError, AttributeError):
                pass  # skip individual unparseable assertions
    return results


def _parse_assert(test: ast.expr) -> dict:
    if isinstance(test, ast.Compare):  # candidate(...) == expected
        call, expected_node = test.left, test.comparators[0]
    elif isinstance(test, ast.Call):  # is_same_list/tree(candidate(...), node)
        call, expected_node = test.args[0], test.args[1]
    else:
        raise ValueError(f"unexpected assert shape: {ast.dump(test)}")

    inputs = [_value(a) for a in call.args] + [_value(kw.value) for kw in call.keywords]
    return {"input": inputs, "expected": _value(expected_node)}


def _value(node: ast.expr) -> object:
    """Evaluate a literal node, unwrapping list_node([...]) / tree_node([...]) to its array."""
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _NODE_WRAPPERS:
        node = node.args[0]
    if isinstance(node, ast.Name) and node.id in _KNOWN_NAMES:
        return _KNOWN_NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Name):
        val = _KNOWN_NAMES.get(node.operand.id)
        if val is not None:
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return +val
    return ast.literal_eval(node)
