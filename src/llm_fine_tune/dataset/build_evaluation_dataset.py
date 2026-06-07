"""Build the evaluation Parquet dataset (Stage 4, step 1).

Reads the base dataset, re-derives the held-out test split (same parameters as
the instruct build), then produces one bigcode_task_payload per held-out
code_snippet × directed language pair. Each row carries a target-language
execution_engine (built from the snippet's expected_input_output_pairs) and all
fields bigcode needs for generation and grading.

Run with: uv run build-evaluation-dataset
Requires: output/leetcode-solutions.parquet (run `make base` first)
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import random

import polars as pl

from llm_fine_tune import loaders
from llm_fine_tune.dataset import splits
from llm_fine_tune.dataset.build_instruct_dataset import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRAC,
    INSTRUCT_LANGUAGES,
)
from llm_fine_tune.dataset.instruction_generator import generate_instruction

BASE_PARQUET_PATH = loaders.OUTPUT_DIR / "leetcode-solutions.parquet"
OUTPUT_PATH = loaders.OUTPUT_DIR / "leetcode-evaluation.parquet"

DEFAULT_SEED = 0

_SCHEMA = {
    "code_snippet_id": pl.Int64,
    "source_language": pl.Utf8,
    "target_language": pl.Utf8,
    "user_prompt": pl.Utf8,
    "code_snippet_to_translate": pl.Utf8,
    "expected_code_snippet_translation": pl.Utf8,
    "execution_engine": pl.Utf8,
    "expected_input_output_pairs": pl.Utf8,
    "difficulty": pl.Utf8,
}


class UnparseableInputOutputPairs(ValueError):
    pass


class UnsupportedInputOutputValue(ValueError):
    pass


def main() -> None:
    args = _parse_args()
    loaders.require_file(BASE_PARQUET_PATH, "run `make base` first.")

    base = pl.read_parquet(BASE_PARQUET_PATH)
    print(f"Loaded base dataset: {base.height:,} code snippets")

    held_out_ids = _held_out_code_snippet_ids(base)
    held_out = base.filter(pl.col("code_snippet_id").is_in(list(held_out_ids)))
    print(
        f"Held-out split (test_frac={DEFAULT_TEST_FRAC}, seed={DEFAULT_SPLIT_SEED}): "
        f"{held_out.height:,} code snippets"
    )

    instruction_rng = random.Random(args.seed)
    print("Building bigcode_task_payloads ...")
    payloads, coverage = _build_bigcode_task_payloads(held_out, instruction_rng)

    loaders.write_parquet(pl.DataFrame(payloads, schema=_SCHEMA), OUTPUT_PATH)
    _print_coverage(coverage)


# ---- Argument parsing ----


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the evaluation Parquet dataset from the base dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for instruction template selection (default: %(default)s).",
    )
    return parser.parse_args()


# ---- Split reproduction ----


def _held_out_code_snippet_ids(base: pl.DataFrame) -> set[int]:
    """Reproduce the instruct split exactly: eligible snippets have solutions in ≥2 languages."""
    eligible_mask = (
        sum(pl.col(lang).is_not_null().cast(pl.Int32) for lang in INSTRUCT_LANGUAGES)
        >= 2
    )
    eligible = base.filter(eligible_mask)
    _, test_side = splits.split_by_key(
        eligible, "code_snippet_id", DEFAULT_TEST_FRAC, DEFAULT_SPLIT_SEED
    )
    return set(test_side["code_snippet_id"].to_list())


# ---- Payload construction ----


def _build_bigcode_task_payloads(
    held_out: pl.DataFrame, instruction_rng: random.Random
) -> tuple[list[dict], dict]:
    payloads: list[dict] = []
    unparseable = 0
    unsupported = 0

    for code_snippet in held_out.iter_rows(named=True):
        for source_language, target_language in itertools.permutations(
            INSTRUCT_LANGUAGES, 2
        ):
            if not (code_snippet[source_language] and code_snippet[target_language]):
                continue
            try:
                expected_input_output_pairs = _parse_input_output_pairs(
                    code_snippet["input_output"]
                )
                execution_engine = _build_execution_engine(
                    code_snippet, target_language, expected_input_output_pairs
                )
            except UnparseableInputOutputPairs:
                unparseable += 1
                continue
            except UnsupportedInputOutputValue:
                unsupported += 1
                continue
            payloads.append(
                _bigcode_task_payload(
                    code_snippet,
                    source_language,
                    target_language,
                    expected_input_output_pairs,
                    execution_engine,
                    instruction_rng,
                )
            )

    coverage = {
        "kept": len(payloads),
        "unparseable": unparseable,
        "unsupported": unsupported,
    }
    return payloads, coverage


def _bigcode_task_payload(
    code_snippet: dict,
    source_language: str,
    target_language: str,
    expected_input_output_pairs: list[dict],
    execution_engine: str,
    instruction_rng: random.Random,
) -> dict:
    return {
        "code_snippet_id": code_snippet["code_snippet_id"],
        "source_language": source_language,
        "target_language": target_language,
        "user_prompt": generate_instruction(
            source_language, target_language, instruction_rng
        ),
        "code_snippet_to_translate": code_snippet[source_language],
        "expected_code_snippet_translation": code_snippet[target_language],
        "execution_engine": execution_engine,
        "expected_input_output_pairs": json.dumps(expected_input_output_pairs),
        "difficulty": code_snippet.get("difficulty"),
    }


# ---- Input/output pair parsing ----


def _parse_input_output_pairs(raw: str | None) -> list[dict]:
    """Parse the raw input_output column into [{"input": [...args], "expected": value}, ...].

    Raises UnparseableInputOutputPairs for null, malformed, or structurally unexpected data.
    """
    if raw is None:
        raise UnparseableInputOutputPairs("input_output is null")

    parsed: object
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise UnparseableInputOutputPairs(f"Cannot parse: {exc}") from exc

    if not isinstance(parsed, dict):
        raise UnparseableInputOutputPairs(
            f"Expected dict, got {type(parsed).__name__}: {str(raw)[:80]}"
        )

    inputs_raw = parsed.get("input") or parsed.get("inputs")
    outputs_raw = parsed.get("output") or parsed.get("outputs")

    if inputs_raw is None or outputs_raw is None:
        raise UnparseableInputOutputPairs(
            f"Missing input/output keys: {list(parsed.keys())}"
        )

    # Normalize to lists of test cases
    if not isinstance(inputs_raw, list):
        inputs_raw = [inputs_raw]
    if not isinstance(outputs_raw, list):
        outputs_raw = [outputs_raw]

    if len(inputs_raw) != len(outputs_raw):
        raise UnparseableInputOutputPairs(
            f"Length mismatch: {len(inputs_raw)} inputs vs {len(outputs_raw)} outputs"
        )

    # Each input element must be a list of arguments for that test case.
    # If it's not a list (single-arg function), wrap it.
    return [
        {
            "input": inp if isinstance(inp, list) else [inp],
            "expected": out,
        }
        for inp, out in zip(inputs_raw, outputs_raw)
    ]


# ---- Execution engine construction ----


def _build_execution_engine(
    code_snippet: dict, target_language: str, pairs: list[dict]
) -> str:
    """Build the target-language driver that runs a translation against its input_output_pairs.

    The engine is combined with the model's code (the Solution class) in execution.assemble_executable.
    It outputs one line per test case: "OK" if the result matches expected, "FAIL" otherwise.
    Raises UnsupportedInputOutputValue for values that cannot be expressed in the target language.
    """
    entry_point = code_snippet.get("entry_point") or "solve"
    if target_language == "python":
        return _python_engine(entry_point, pairs)
    elif target_language == "cpp":
        return _cpp_engine(entry_point, pairs)
    elif target_language == "java":
        return _java_engine(entry_point, pairs)
    else:
        raise UnsupportedInputOutputValue(
            f"Unsupported target language: {target_language!r}"
        )


def _python_engine(entry_point: str, pairs: list[dict]) -> str:
    cases_repr = repr(pairs)
    return (
        "\n"
        "# === EXECUTION ENGINE ===\n"
        f"_CASES = {cases_repr}\n"
        "\n"
        "_sol = Solution()\n"
        "for _case in _CASES:\n"
        "    try:\n"
        f"        _actual = _sol.{entry_point}(*_case['input'])\n"
        "        print('OK' if _actual == _case['expected'] else 'FAIL')\n"
        "    except Exception:\n"
        "        print('FAIL')\n"
    )


def _cpp_engine(entry_point: str, pairs: list[dict]) -> str:
    test_blocks: list[str] = []
    for i, pair in enumerate(pairs):
        input_args: list[str] = []
        arg_names: list[str] = []
        for j, arg in enumerate(pair["input"]):
            cpp_type = _cpp_type(arg)
            cpp_lit = _cpp_literal(arg)
            var_name = f"_arg_{i}_{j}"
            input_args.append(f"        {cpp_type} {var_name} = {cpp_lit};")
            arg_names.append(var_name)
        expected_lit = _cpp_literal(pair["expected"])
        call = f"s.{entry_point}({', '.join(arg_names)})"
        block = (
            f"    // test case {i}\n"
            "    {\n"
            + "\n".join(input_args)
            + f"\n        auto _expected_{i} = {expected_lit};\n"
            f"        auto _result_{i} = {call};\n"
            f'        std::cout << (_result_{i} == _expected_{i} ? "OK" : "FAIL") << "\\n";\n'
            "    }"
        )
        test_blocks.append(block)

    body = "\n".join(test_blocks)
    return (
        "\n"
        "// === EXECUTION ENGINE ===\n"
        "int main() {\n"
        "    Solution s;\n"
        f"{body}\n"
        "    return 0;\n"
        "}\n"
    )


def _java_engine(entry_point: str, pairs: list[dict]) -> str:
    test_blocks: list[str] = []
    for i, pair in enumerate(pairs):
        input_args: list[str] = []
        arg_names: list[str] = []
        for j, arg in enumerate(pair["input"]):
            java_type = _java_type(arg)
            java_lit = _java_literal(arg)
            var_name = f"_arg{i}_{j}"
            input_args.append(f"            {java_type} {var_name} = {java_lit};")
            arg_names.append(var_name)
        expected_lit = _java_literal(pair["expected"])
        expected_type = _java_type(pair["expected"])
        call = f"_s.{entry_point}({', '.join(arg_names)})"
        block = (
            f"        // test case {i}\n"
            "        {\n"
            + "\n".join(input_args)
            + f"\n            {expected_type} _expected{i} = {expected_lit};\n"
            f"            Object _result{i} = {call};\n"
            f'            System.out.println(_eq(_result{i}, _expected{i}) ? "OK" : "FAIL");\n'
            "        }"
        )
        test_blocks.append(block)

    body = "\n".join(test_blocks)
    return (
        "\n"
        "// === EXECUTION ENGINE ===\n"
        "class Main {\n"
        "    static boolean _eq(Object a, Object b) {\n"
        "        if (a == b) return true;\n"
        "        if (a == null || b == null) return false;\n"
        "        if (a instanceof int[] && b instanceof int[])\n"
        "            return java.util.Arrays.equals((int[]) a, (int[]) b);\n"
        "        if (a instanceof boolean[] && b instanceof boolean[])\n"
        "            return java.util.Arrays.equals((boolean[]) a, (boolean[]) b);\n"
        "        if (a instanceof double[] && b instanceof double[])\n"
        "            return java.util.Arrays.equals((double[]) a, (double[]) b);\n"
        "        if (a instanceof String[] && b instanceof String[])\n"
        "            return java.util.Arrays.equals((String[]) a, (String[]) b);\n"
        "        if (a instanceof int[][] && b instanceof int[][])\n"
        "            return java.util.Arrays.deepEquals((int[][]) a, (int[][]) b);\n"
        "        return java.util.Objects.equals(a, b);\n"
        "    }\n"
        "\n"
        "    public static void main(String[] _args) {\n"
        "        Solution _s = new Solution();\n"
        f"{body}\n"
        "    }\n"
        "}\n"
    )


# ---- Language-specific type and literal generators ----


def _cpp_type(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "std::string"
    if isinstance(value, list):
        if not value:
            return "std::vector<int>"
        elem = value[0]
        return f"std::vector<{_cpp_type(elem)}>"
    raise UnsupportedInputOutputValue(
        f"No C++ type mapping for {type(value).__name__!r}: {value!r}"
    )


def _cpp_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, list):
        elements = ", ".join(_cpp_literal(v) for v in value)
        return "{" + elements + "}"
    raise UnsupportedInputOutputValue(
        f"No C++ literal for {type(value).__name__!r}: {value!r}"
    )


def _java_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "String"
    if isinstance(value, list):
        if not value:
            return "int[]"
        elem = value[0]
        elem_type = _java_type(elem)
        primitive_arrays = {
            "int": "int[]",
            "boolean": "boolean[]",
            "double": "double[]",
        }
        return primitive_arrays.get(elem_type, f"{elem_type}[]")
    raise UnsupportedInputOutputValue(
        f"No Java type mapping for {type(value).__name__!r}: {value!r}"
    )


def _java_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, list):
        if not value:
            return "new int[]{}"
        elem = value[0]
        elem_type = _java_type(elem)
        elements = ", ".join(_java_literal(v) for v in value)
        return f"new {elem_type}{{{elements}}}"
    raise UnsupportedInputOutputValue(
        f"No Java literal for {type(value).__name__!r}: {value!r}"
    )


# ---- Reporting ----


def _print_coverage(coverage: dict) -> None:
    total = coverage["kept"] + coverage["unparseable"] + coverage["unsupported"]
    print(
        f"\nEvaluation dataset built:"
        f"\n  kept:        {coverage['kept']:,} bigcode_task_payloads → {OUTPUT_PATH}"
        f"\n  unparseable: {coverage['unparseable']:,} rows skipped (bad input_output)"
        f"\n  unsupported: {coverage['unsupported']:,} rows skipped (unsupported value types)"
        f"\n  total:       {total:,} candidate rows"
    )


if __name__ == "__main__":
    main()
