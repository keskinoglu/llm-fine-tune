"""Structural tests for MultiPL-E program assembly (run_code_benchmark_scoring).

These check the assembled source is well-formed (balanced braces, no duplicated signature) without
compiling — the host has no javac. Actual compilation is confirmed by the cluster shakeout.
"""

from __future__ import annotations

from llm_fine_tune.evaluation.run_code_benchmark_scoring import (
    _assemble_multipl_e_program,
    _java_method_source,
)

# Shapes mirror real nuprl/MultiPL-E humaneval-java rows: prompt opens the class + method, tests
# begin by closing the method and end by closing the class.
JAVA_PROMPT = (
    "import java.util.*;\n"
    "import org.javatuples.*;\n"
    "class Problem {\n"
    "    // Check if any two numbers are closer than threshold.\n"
    "    public static boolean hasCloseElements(ArrayList<Float> numbers, float threshold) {\n"
)
JAVA_TESTS = (
    "    }\n"
    "    public static void main(String[] args) {\n"
    "    assert(hasCloseElements((new ArrayList<Float>(Arrays.asList((float)1.0f))), (0.3f)) == (false));\n"
    "    }\n"
    "}\n"
)
_SIGNATURE = "hasCloseElements(ArrayList<Float> numbers, float threshold)"


def _braces_balanced(source: str) -> bool:
    depth = 0
    for ch in source:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def test_java_bare_method_assembly_is_wellformed():
    completion = (
        "public static boolean hasCloseElements(ArrayList<Float> numbers, float threshold) {\n"
        "    for (int i = 0; i < numbers.size(); i++)\n"
        "        for (int j = i + 1; j < numbers.size(); j++)\n"
        "            if (Math.abs(numbers.get(i) - numbers.get(j)) < threshold) return true;\n"
        "    return false;\n"
        "}"
    )
    program = _assemble_multipl_e_program(completion, JAVA_TESTS, "java", JAVA_PROMPT)
    assert _braces_balanced(program)
    assert program.count("class Problem {") == 1
    assert program.count("public static void main") == 1
    assert program.count(_SIGNATURE) == 1  # no duplicated signature
    assert "import org.javatuples.*;" in program  # prompt imports carried over


def test_java_full_class_completion_is_not_duplicated():
    # The model wraps its method in its own class — the case that scored java ~0 (signature appeared
    # twice: once from the prompt's open method, once from the model's). The fix unwraps it.
    completion = (
        "class Problem {\n"
        "  public static boolean hasCloseElements(ArrayList<Float> numbers, float threshold) {\n"
        "    return false;\n"
        "  }\n"
        "}"
    )
    program = _assemble_multipl_e_program(completion, JAVA_TESTS, "java", JAVA_PROMPT)
    assert _braces_balanced(program)
    assert program.count(_SIGNATURE) == 1
    assert program.count("public static void main") == 1


def test_java_method_source_unwraps_a_returned_class():
    completion = "class Solution {\n  public static int f() { return 1; }\n}"
    extracted = _java_method_source(completion)
    assert "class" not in extracted
    assert "public static int f()" in extracted


def test_java_method_source_passes_through_a_bare_method():
    completion = "public static int f() { return 1; }"
    assert _java_method_source(completion) == completion


# ---------------------------------------------------------------------------
# Rust — same shape as C++ (prompt opens a free fn, tests close it + a main of asserts)
# ---------------------------------------------------------------------------

RUST_PROMPT = (
    "/// Check if any two numbers are closer than threshold.\n"
    "fn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool {\n"
)
RUST_TESTS = (
    "}\n\n"
    "fn main() {\n"
    "    let candidate = has_close_elements;\n"
    "    assert_eq!(candidate(vec![1.0, 2.0, 3.9], 0.3), true);\n"
    "}\n"
)
_RUST_SIG = "fn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool"


def test_rust_assembly_is_wellformed():
    completion = (
        "fn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool {\n"
        "    for i in 0..numbers.len() {\n"
        "        for j in (i + 1)..numbers.len() {\n"
        "            if (numbers[i] - numbers[j]).abs() < threshold { return true; }\n"
        "        }\n"
        "    }\n"
        "    false\n"
        "}"
    )
    program = _assemble_multipl_e_program(completion, RUST_TESTS, "rust", RUST_PROMPT)
    assert _braces_balanced(program)
    assert (
        program.count(_RUST_SIG) == 1
    )  # signature from the prompt, body sliced from completion
    assert "fn main()" in program


# ---------------------------------------------------------------------------
# Go — testing framework; prompt carries package + imports; tests don't close the fn
# ---------------------------------------------------------------------------

GO_PROMPT = (
    "package has_close_elements_test\n\n"
    "import (\n"
    '    "testing"\n'
    '    "math"\n'
    ")\n\n"
    "// Check if any two numbers are closer than threshold.\n"
    "func has_close_elements(numbers []float64, threshold float64) bool {\n"
)
GO_TESTS = (
    "func TestHasCloseElements(t *testing.T) {\n"
    "    candidate := has_close_elements\n"
    '    if candidate([]float64{1.0, 2.0}, 0.3) != false { t.Error("fail") }\n'
    "}\n"
)
_GO_SIG = "func has_close_elements(numbers []float64, threshold float64) bool"


def test_go_assembly_is_wellformed():
    completion = (
        "func has_close_elements(numbers []float64, threshold float64) bool {\n"
        "    for i := 0; i < len(numbers); i++ {\n"
        "        for j := i + 1; j < len(numbers); j++ {\n"
        "            if math.Abs(numbers[i]-numbers[j]) < threshold { return true }\n"
        "        }\n"
        "    }\n"
        "    return false\n"
        "}"
    )
    program = _assemble_multipl_e_program(completion, GO_TESTS, "go", GO_PROMPT)
    assert _braces_balanced(program)
    assert "package has_close_elements_test" in program
    assert "import (" in program
    assert (
        program.count(_GO_SIG) == 1
    )  # from the model's completion, prompt's open sig dropped
    assert "func TestHasCloseElements" in program
