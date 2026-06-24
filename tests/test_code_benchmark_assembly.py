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
