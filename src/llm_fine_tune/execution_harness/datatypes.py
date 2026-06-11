"""The Python datatypes we can express as target_language (cpp/java/python) code.

When building an execution_engine: ints, floats, strings, lists of them, and
LeetCode's ListNode/TreeNode. Each function RETURNS target_language source code
— a type name, a literal, a node's class definitions, or the code to build a
node from its array / read one back.  A value of an unsupported datatype raises
UnsupportedInputOutputValue, which the build catches to skip just that
input_output_pair.

Imported by the dataset build (to generate execution_engine code) and by
execution (to prepend the node definitions).
"""

from __future__ import annotations

import math
import re

# What a parameter / return value is.
LIST_NODE = "list_node"
TREE_NODE = "tree_node"
PLAIN = "plain"


class UnsupportedInputOutputValue(ValueError):
    """Raised when a value cannot be expressed as target-language code."""


# ---------------------------------------------------------------------------
# Scalar datatype → target_language type name + literal
# ---------------------------------------------------------------------------


def cpp_type(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int" if -(2**31) <= value <= 2**31 - 1 else "long long"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "std::string"
    if isinstance(value, list):
        if not value:
            return "std::vector<int>"
        return f"std::vector<{cpp_type(value[0])}>"
    raise UnsupportedInputOutputValue(
        f"No C++ type mapping for {type(value).__name__!r}: {value!r}"
    )


def cpp_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value) if -(2**31) <= value <= 2**31 - 1 else f"{value}LL"
    if isinstance(value, float):
        if math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"
        if math.isnan(value):
            return "NAN"
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, list):
        elements = ", ".join(cpp_literal(v) for v in value)
        return "{" + elements + "}"
    raise UnsupportedInputOutputValue(
        f"No C++ literal for {type(value).__name__!r}: {value!r}"
    )


def java_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int" if -(2**31) <= value <= 2**31 - 1 else "long"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "String"
    if isinstance(value, list):
        if not value:
            return "int[]"
        elem_type = java_type(value[0])
        primitive_arrays = {
            "int": "int[]",
            "boolean": "boolean[]",
            "double": "double[]",
        }
        return primitive_arrays.get(elem_type, f"{elem_type}[]")
    raise UnsupportedInputOutputValue(
        f"No Java type mapping for {type(value).__name__!r}: {value!r}"
    )


def java_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value) if -(2**31) <= value <= 2**31 - 1 else f"{value}L"
    if isinstance(value, float):
        if math.isinf(value):
            return (
                "Double.POSITIVE_INFINITY" if value > 0 else "Double.NEGATIVE_INFINITY"
            )
        if math.isnan(value):
            return "Double.NaN"
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, list):
        if not value:
            return "new int[]{}"
        elem_type = java_type(value[0])
        elements = ", ".join(java_literal(v) for v in value)
        return f"new {elem_type}[]{{{elements}}}"
    raise UnsupportedInputOutputValue(
        f"No Java literal for {type(value).__name__!r}: {value!r}"
    )


def python_literal(value: object) -> str:
    """Return a Python literal string for *value*.

    Uses repr() for plain types. Raises UnsupportedInputOutputValue for
    values that cannot be expressed as safe Python code.
    """
    if isinstance(value, (bool, int, float, str, list)):
        return repr(value)
    raise UnsupportedInputOutputValue(
        f"No Python literal for {type(value).__name__!r}: {value!r}"
    )


# ---------------------------------------------------------------------------
# Node datatype helpers
# ---------------------------------------------------------------------------


def node_definitions(target_language: str) -> str:
    """Return ListNode + TreeNode class definitions for *target_language*.

    Also includes the helpers that read a node back to its array text.
    Prepended before the model's code so it and the execution_engine can use
    the types.
    """
    if target_language == "python":
        return _PYTHON_NODE_DEFS
    if target_language == "cpp":
        return _CPP_NODE_DEFS
    if target_language == "java":
        return _JAVA_NODE_DEFS
    raise UnsupportedInputOutputValue(f"No node definitions for {target_language!r}")


_PYTHON_NODE_DEFS = """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        return isinstance(other, ListNode) and self.val == other.val and self.next == other.next

    def __repr__(self):
        vals = []
        cur = self
        while cur:
            vals.append(str(cur.val))
            cur = cur.next
        return "->".join(vals)

    @staticmethod
    def from_array(arr):
        if not arr:
            return None
        head = ListNode(arr[0])
        cur = head
        for v in arr[1:]:
            cur.next = ListNode(v)
            cur = cur.next
        return head

    def to_array(self):
        vals = []
        cur = self
        while cur:
            vals.append(cur.val)
            cur = cur.next
        return vals


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return self.val == other.val and self.left == other.left and self.right == other.right

    def __repr__(self):
        return f"TreeNode({self.val})"

    @staticmethod
    def from_array(arr):
        if not arr or arr[0] is None:
            return None
        root = TreeNode(arr[0])
        queue = [root]
        i = 1
        while queue and i < len(arr):
            node = queue.pop(0)
            if arr[i] is not None:
                node.left = TreeNode(arr[i])
                queue.append(node.left)
            i += 1
            if i < len(arr):
                if arr[i] is not None:
                    node.right = TreeNode(arr[i])
                    queue.append(node.right)
                i += 1
        return root

    def to_array(self):
        if not self:
            return []
        result = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node is None:
                result.append(None)
            else:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
        # Trim trailing nulls
        while result and result[-1] is None:
            result.pop()
        return result
"""

_CPP_NODE_DEFS = """
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

static std::vector<int> _listNodeToArray(ListNode* head) {
    std::vector<int> r;
    while (head) { r.push_back(head->val); head = head->next; }
    return r;
}

static std::vector<std::optional<int>> _treeNodeToArray(TreeNode* root) {
    if (!root) return {};
    std::vector<std::optional<int>> r;
    std::queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        TreeNode* n = q.front(); q.pop();
        if (n) { r.push_back(n->val); q.push(n->left); q.push(n->right); }
        else   { r.push_back(std::nullopt); }
    }
    while (!r.empty() && !r.back()) r.pop_back();
    return r;
}

static ListNode* _arrayToListNode(const std::vector<int>& arr) {
    ListNode* head = nullptr; ListNode* cur = nullptr;
    for (int v : arr) { ListNode* n = new ListNode(v); if (!head) head = n; else cur->next = n; cur = n; }
    return head;
}
static TreeNode* _arrayToTreeNode(const std::vector<std::optional<int>>& arr) {
    if (arr.empty() || !arr[0].has_value()) return nullptr;
    TreeNode* root = new TreeNode(arr[0].value());
    std::queue<TreeNode*> q; q.push(root); size_t i = 1;
    while (!q.empty() && i < arr.size()) {
        TreeNode* node = q.front(); q.pop();
        if (i < arr.size()) { if (arr[i].has_value()) { node->left  = new TreeNode(arr[i].value()); q.push(node->left);  } i++; }
        if (i < arr.size()) { if (arr[i].has_value()) { node->right = new TreeNode(arr[i].value()); q.push(node->right); } i++; }
    }
    return root;
}
"""

_JAVA_NODE_DEFS = """
class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class _NodeHelpers {
    static int[] listNodeToArray(ListNode head) {
        java.util.List<Integer> r = new java.util.ArrayList<>();
        while (head != null) { r.add(head.val); head = head.next; }
        return r.stream().mapToInt(i -> i).toArray();
    }
    static Integer[] treeNodeToArray(TreeNode root) {
        if (root == null) return new Integer[0];
        java.util.Queue<TreeNode> q = new java.util.LinkedList<>();
        java.util.List<Integer> r = new java.util.ArrayList<>();
        q.add(root);
        while (!q.isEmpty()) {
            TreeNode n = q.poll();
            if (n != null) { r.add(n.val); q.add(n.left); q.add(n.right); }
            else r.add(null);
        }
        int last = r.size() - 1;
        while (last >= 0 && r.get(last) == null) last--;
        return r.subList(0, last + 1).toArray(new Integer[0]);
    }
    static ListNode arrayToListNode(int[] arr) {
        ListNode head = null, cur = null;
        for (int v : arr) { ListNode n = new ListNode(v); if (head == null) head = n; else cur.next = n; cur = n; }
        return head;
    }
    static TreeNode arrayToTreeNode(Integer[] arr) {
        if (arr.length == 0 || arr[0] == null) return null;
        TreeNode root = new TreeNode(arr[0]);
        java.util.Queue<TreeNode> q = new java.util.LinkedList<>(); q.add(root); int i = 1;
        while (!q.isEmpty() && i < arr.length) {
            TreeNode node = q.poll();
            if (i < arr.length) { if (arr[i] != null) { node.left  = new TreeNode(arr[i]); q.add(node.left);  } i++; }
            if (i < arr.length) { if (arr[i] != null) { node.right = new TreeNode(arr[i]); q.add(node.right); } i++; }
        }
        return root;
    }
}
"""


def list_node_array_text(array_values: list) -> str:
    """Return the canonical text form of a ListNode array: ``[1, 2, 3]``.

    The build bakes this in as the expected comparison; the read-back helper
    in the execution_engine matches it.
    """
    return repr(array_values)


def tree_node_array_text(level_order_array_values: list) -> str:
    """Return the canonical text form of a TreeNode level-order array with trailing nulls trimmed.

    ``[1, null, 2]``  — trailing nulls removed so the comparison is stable.
    """
    trimmed = list(level_order_array_values)
    while trimmed and trimmed[-1] is None:
        trimmed.pop()
    return repr(trimmed)


# ---------------------------------------------------------------------------
# Node type detection from Python reference signature
# ---------------------------------------------------------------------------

_NODE_TYPE_HINTS: dict[str, str] = {
    "ListNode": LIST_NODE,
    "TreeNode": TREE_NODE,
}

_OPTIONAL_PATTERN = re.compile(
    r"^(?:Optional\s*\[\s*(\w+)\s*\]|(\w+)\s*(?:\|\s*None|None\s*\|)\s*(\w*))$"
)


def extract_type_hints(code_snippet: dict) -> dict:
    """Extract raw Python type hint strings from the entry_point function.

    Returns ``{"parameter_hints": [str, ...], "return_hint": str}``.
    Each element is the raw annotation text (e.g. ``"List[int]"``, ``"int"``).
    Empty string means no annotation was present for that position.
    """
    entry_point = code_snippet.get("entry_point") or "solve"
    python_code = code_snippet.get("python")
    if not python_code:
        return {"parameter_hints": [], "return_hint": ""}

    func_name = entry_point.split(".")[-1] if "." in entry_point else entry_point
    if func_name in ("None", ""):
        func_name = "solve"

    func_match = re.search(
        rf"def\s+{re.escape(func_name)}\s*\((.*?)\)\s*(->\s*(.*?)\s*)?:",
        python_code,
        re.DOTALL,
    )
    if not func_match:
        return {"parameter_hints": [], "return_hint": ""}

    param_section = func_match.group(1)
    return_hint = (func_match.group(3) or "").strip()

    hints: list[str] = []
    for p in param_section.split(","):
        p = p.strip()
        if p in ("self", "") or p.startswith("*"):
            continue
        if ":" in p:
            _, raw = p.split(":", 1)
            # Strip default value (e.g. "int = 0" → "int")
            hints.append(raw.split("=")[0].strip())
        else:
            hints.append("")

    return {"parameter_hints": hints, "return_hint": return_hint}


def detect_node_types(code_snippet: dict) -> dict:
    """Read the *Python reference signature* and return node type info.

    Returns ``{"parameters": [LIST_NODE|TREE_NODE|PLAIN, ...], "return_value": ...,
    "parameter_hints": [str, ...], "return_hint": str}``
    for the ``entry_point`` function.  PLAIN = a plain value rendered as a literal.
    Raises ``UnsupportedInputOutputValue`` if no Python reference solution exists.
    """
    python_code = code_snippet.get("python")
    if not python_code:
        raise UnsupportedInputOutputValue("No Python reference solution")

    type_info = extract_type_hints(code_snippet)
    param_hints = type_info["parameter_hints"]
    return_hint = type_info["return_hint"]

    param_types = [_classify_type_hint(h) for h in param_hints]
    return_type = _classify_type_hint(return_hint) if return_hint else PLAIN

    return {
        "parameters": param_types,
        "return_value": return_type,
        "parameter_hints": param_hints,
        "return_hint": return_hint,
    }


def _classify_type_hint(hint: str) -> str:
    """Classify a single type hint string → LIST_NODE / TREE_NODE / PLAIN."""
    hint = hint.strip()
    # Remove list[...] / List[...] / Sequence[...] / tuple[...] wrappers
    list_match = re.match(r"(?:[Ll]ist|Sequence|[Tt]uple)\s*\[\s*(.*)\s*\]", hint)
    if list_match:
        return _classify_type_hint(list_match.group(1))

    # Remove Optional[...] wrappers and X | None patterns
    opt_match = _OPTIONAL_PATTERN.match(hint)
    if opt_match:
        inner = opt_match.group(1) or opt_match.group(2) or opt_match.group(3)
        inner = inner.strip()
        if inner in _NODE_TYPE_HINTS:
            return _NODE_TYPE_HINTS[inner]
        return _classify_type_hint(inner)

    # Direct type name
    if hint in _NODE_TYPE_HINTS:
        return _NODE_TYPE_HINTS[hint]

    # Skip self or empty
    if hint in ("", "self"):
        return PLAIN

    # Anything else we don't recognize is assumed plain
    return PLAIN


# ---------------------------------------------------------------------------
# Target-language type derivation from Python type hints + Java signature
# ---------------------------------------------------------------------------


def cpp_type_from_hint(
    hint: str, sample_value: object = None, wide: bool = False
) -> str:
    """Return the C++ type name for a Python type annotation.

    Uses the hint for all structural information (nesting, element type).
    Falls back to cpp_type(sample_value) only when the hint is absent or
    unrecognised.  *wide* forces ``int`` → ``long long`` regardless of sample value.
    """
    hint = (hint or "").strip()

    # list[X] / List[X] / Sequence[X] / tuple[X] → std::vector<inner>
    m = re.match(r"^(?:[Ll]ist|Sequence|[Tt]uple)\s*\[\s*(.*)\s*\]$", hint, re.DOTALL)
    if m:
        inner = cpp_type_from_hint(m.group(1).strip(), _first_elem(sample_value), wide)
        return f"std::vector<{inner}>"

    # Optional[X] / X | None → unwrap
    opt_m = _OPTIONAL_PATTERN.match(hint)
    if opt_m:
        inner = (opt_m.group(1) or opt_m.group(2) or opt_m.group(3)).strip()
        return cpp_type_from_hint(inner, sample_value, wide)

    if hint == "ListNode":
        return "ListNode*"
    if hint == "TreeNode":
        return "TreeNode*"

    if hint == "int":
        return "long long" if (wide or needs_int64(sample_value)) else "int"
    if hint in ("float", "double"):
        return "double"
    if hint in ("str", "String"):
        return "std::string"
    if hint == "bool":
        return "bool"
    if hint == "char":
        return "char"

    # Unknown hint → value-based fallback
    return cpp_type(sample_value) if sample_value is not None else "int"


def _first_elem(value: object) -> object:
    return value[0] if isinstance(value, list) and value else None


def needs_int64(value: object) -> bool:
    """True if any integer anywhere in (possibly nested) *value* exceeds 32-bit signed range."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return not (-(2**31) <= value <= 2**31 - 1)
    if isinstance(value, list):
        return any(needs_int64(v) for v in value)
    return False


def parse_java_param_types(java_source: str, func_name: str) -> list[str] | None:
    """Extract Java parameter type strings from the method signature in *java_source*.

    Returns a list like ``["List<Integer>", "int"]`` or ``None`` if the
    signature cannot be found.  Handles generic types (``List<List<Integer>>``)
    by splitting on ``,`` at ``< >`` depth 0.
    """
    if not java_source or not func_name:
        return None

    m = re.search(rf"\b{re.escape(func_name)}\s*\(([^)]*)\)", java_source)
    if not m:
        return None

    params_str = m.group(1).strip()
    if not params_str:
        return []

    types: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params_str:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            t = _java_type_from_param_decl("".join(current).strip())
            if t:
                types.append(t)
            current = []
        else:
            current.append(ch)
    if current:
        t = _java_type_from_param_decl("".join(current).strip())
        if t:
            types.append(t)

    return types or None


def _java_type_from_param_decl(decl: str) -> str | None:
    """Extract the type from a Java param declaration (``"List<Integer> nums"`` → ``"List<Integer>"``).

    Strips ``final``, handles varargs (``int...`` → ``int[]``).
    """
    decl = re.sub(r"\bfinal\b", "", decl).strip()
    decl = decl.replace("...", "[]")
    parts = decl.rsplit(None, 1)
    return parts[0].strip() if len(parts) == 2 else None


def widen_java_int_type(java_type_str: str) -> str:
    """Widen int→long for a Java type when values need 64 bits."""
    return {
        "int": "long",
        "int[]": "long[]",
        "int[][]": "long[][]",
        "Integer": "Long",
        "List<Integer>": "List<Long>",
        "List<List<Integer>>": "List<List<Long>>",
    }.get(java_type_str.strip(), java_type_str)


def _java_scalar_lit(value: object, base_type: str) -> str:
    """One scalar literal typed as *base_type*; adds L suffix for long, char literal for char."""
    if (
        base_type in ("long", "Long")
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return f"{value}L"
    if (
        base_type in ("char", "Character")
        and isinstance(value, str)
        and len(value) == 1
    ):
        ch = value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        return f"'{ch}'"
    return java_literal(value)


def java_literal_for_type(value: object, java_type_str: str) -> str:
    """Build a Java literal for *value* declared as *java_type_str*.

    Resolves the ``int[]`` vs ``List<Integer>`` ambiguity that value-inference
    alone cannot.  Falls back to ``java_literal(value)`` for unrecognised types.
    """
    t = java_type_str.strip()
    if t in (
        "int",
        "long",
        "double",
        "float",
        "boolean",
        "char",
        "String",
        "Integer",
        "Long",
        "Double",
        "Float",
        "Boolean",
        "Character",
        "Object",
    ):
        return _java_scalar_lit(value, t)
    m1 = re.match(r"^(int|long|double|float|boolean|char|String)\[\]$", t)
    if m1:
        base = m1.group(1)
        if not isinstance(value, list):
            return f"new {base}[]{{}}"
        return (
            f"new {base}[]{{"
            + ", ".join(_java_scalar_lit(v, base) for v in value)
            + "}"
        )
    m2 = re.match(r"^(int|long|double|boolean|char|String)\[\]\[\]$", t)
    if m2:
        base = m2.group(1)
        if not isinstance(value, list):
            return f"new {base}[][]{{}}"
        rows = ", ".join(
            ("{" + ", ".join(_java_scalar_lit(v, base) for v in row) + "}")
            if isinstance(row, list)
            else "{}"
            for row in value
        )
        return f"new {base}[][]{{{rows}}}"
    m_list = re.match(r"^List<(.+)>$", t)
    if m_list:
        inner = m_list.group(1).strip()
        if not isinstance(value, list) or not value:
            return "new java.util.ArrayList<>()"
        return (
            "new java.util.ArrayList<>(java.util.Arrays.asList("
            + ", ".join(_java_elem_lit(v, inner) for v in value)
            + "))"
        )
    return java_literal(value)


def _java_elem_lit(value: object, inner_type: str) -> str:
    """Literal for one element inside a Java generic collection."""
    if re.match(r"^List<", inner_type):
        return java_literal_for_type(value, inner_type)
    if inner_type == "Long" and isinstance(value, int) and not isinstance(value, bool):
        return f"{value}L"
    if inner_type == "Character" and isinstance(value, str) and len(value) == 1:
        return _java_scalar_lit(value, "char")
    return java_literal(value)


# ---------------------------------------------------------------------------
# Node array literals for C++ and Java engines
# ---------------------------------------------------------------------------


def cpp_optional_int_vector_literal(lst: list) -> str:
    """Return a C++ std::vector<std::optional<int>> initializer for *lst* (None → std::nullopt).

    Trailing Nones are trimmed to match the canonical _treeNodeToArray output.
    """
    trimmed = list(lst)
    while trimmed and trimmed[-1] is None:
        trimmed.pop()
    elements = ["std::nullopt" if v is None else str(v) for v in trimmed]
    return "{" + ", ".join(elements) + "}"


def java_nullable_int_array_literal(lst: list) -> str:
    """Return a Java Integer[] literal for *lst* (None → null).

    Trailing Nones are trimmed to match the canonical _treeNodeToArray output.
    """
    trimmed = list(lst)
    while trimmed and trimmed[-1] is None:
        trimmed.pop()
    elements = ["null" if v is None else str(v) for v in trimmed]
    return "new Integer[]{" + ", ".join(elements) + "}"
