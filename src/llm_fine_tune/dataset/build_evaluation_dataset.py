"""Build the evaluation Parquet dataset (Stage 4, step 1).

Reads the base dataset, re-derives the held-out test split (same parameters as
the instruct build), then produces one bigcode_task_payload per held-out
parallel x directed language pair. Each row carries a target-language
execution_engine (built from the parallel's shared input_output_pairs) and all
fields bigcode needs for generation and grading.

Run with: uv run build-evaluation-dataset
Requires: output/leetcode-solutions.parquet (run `make base` first)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import polars as pl
from tqdm import tqdm

from llm_fine_tune import loaders
from llm_fine_tune.execution_harness import datatypes
from llm_fine_tune.dataset import splits
from llm_fine_tune.dataset.build_instruct_dataset import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRAC,
    INSTRUCT_LANGUAGES,
)
from llm_fine_tune.dataset.instruction_generator import generate_instruction
from llm_fine_tune.dataset.quality import test_assertion_parser

BASE_PARQUET_PATH = loaders.OUTPUT_DIR / "leetcode-solutions.parquet"
OUTPUT_PATH = loaders.OUTPUT_DIR / "leetcode-evaluation.parquet"

DEFAULT_SEED = 0

_EXCLUSIONS_PATH = Path(__file__).parent / "quality" / "exclusions.json"

_SCHEMA = {
    "parallel_id": pl.Int64,
    "source_language": pl.Utf8,
    "target_language": pl.Utf8,
    "user_prompt": pl.Utf8,
    "code_snippet_to_translate": pl.Utf8,
    "expected_code_snippet_translation": pl.Utf8,
    "execution_engine": pl.Utf8,
    "expected_input_output_pairs": pl.Utf8,
    "difficulty": pl.Utf8,
}


def main() -> None:
    args = _parse_args()
    loaders.require_file(BASE_PARQUET_PATH, "run `make base` first.")

    base_dataset = pl.read_parquet(BASE_PARQUET_PATH)
    print(f"Loaded base dataset: {base_dataset.height:,} parallels")

    held_out = base_dataset.filter(
        pl.col("parallel_id").is_in(list(_held_out_parallel_ids(base_dataset)))
    )
    print(
        f"Held-out split (test_frac={DEFAULT_TEST_FRAC}, seed={DEFAULT_SPLIT_SEED}): "
        f"{held_out.height:,} parallels"
    )

    instruction_rng = random.Random(args.seed)
    print("Building bigcode_task_payloads ...")
    bigcode_task_payloads, input_output_pairs_usability_report = (
        _build_bigcode_task_payloads(held_out, instruction_rng)
    )

    loaders.write_parquet(
        pl.DataFrame(bigcode_task_payloads, schema=_SCHEMA), OUTPUT_PATH
    )
    _print_report(input_output_pairs_usability_report)


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


def _held_out_parallel_ids(base: pl.DataFrame) -> set[int]:
    """Reproduce the instruct split exactly: eligible parallels have code_snippets in >=2 languages."""
    eligible_mask = (
        sum(pl.col(lang).is_not_null().cast(pl.Int32) for lang in INSTRUCT_LANGUAGES)
        >= 2
    )
    eligible = base.filter(eligible_mask)
    _, test_side = splits.split_by_key(
        eligible, "parallel_id", DEFAULT_TEST_FRAC, DEFAULT_SPLIT_SEED
    )
    return set(test_side["parallel_id"].to_list())


# ---- Payload construction ----


def _build_bigcode_task_payloads(
    held_out: pl.DataFrame, instruction_rng: random.Random
) -> tuple[list[dict], dict]:
    bigcode_task_payloads: list[dict] = []
    input_output_pairs_usability_report = _new_report()
    exclusions = _load_exclusions()

    for parallel in tqdm(
        held_out.iter_rows(named=True),
        total=held_out.height,
        desc="Building rows",
    ):
        for (
            source_language,
            target_language,
            execution_engine,
            expected_input_output_pairs,
        ) in _executable_translations(
            parallel, input_output_pairs_usability_report, exclusions
        ):
            bigcode_task_payloads.append(
                _build_bigcode_task_payload(
                    parallel,
                    source_language,
                    target_language,
                    expected_input_output_pairs,
                    execution_engine,
                    instruction_rng,
                )
            )

    return bigcode_task_payloads, input_output_pairs_usability_report


def _build_bigcode_task_payload(
    parallel: dict,
    source_language: str,
    target_language: str,
    expected_input_output_pairs: list[dict],
    execution_engine: str,
    instruction_rng: random.Random,
) -> dict:
    """One bigcode_task_payload row (expected_input_output_pairs = json.dumps(...))."""
    return {
        "parallel_id": parallel["parallel_id"],
        "source_language": source_language,
        "target_language": target_language,
        "user_prompt": generate_instruction(
            source_language, target_language, instruction_rng
        ),
        "code_snippet_to_translate": parallel[source_language],
        "expected_code_snippet_translation": parallel[target_language],
        "execution_engine": execution_engine,
        "expected_input_output_pairs": json.dumps(expected_input_output_pairs),
        "difficulty": parallel.get("difficulty"),
    }


def _load_exclusions() -> set[tuple[int, str]]:
    if not _EXCLUSIONS_PATH.exists():
        return set()
    records = json.loads(_EXCLUSIONS_PATH.read_text())
    return {(r["parallel_id"], r["target_language"]) for r in records}


def _executable_translations(
    parallel: dict,
    input_output_pairs_usability_report: dict,
    exclusions: set[tuple[int, str]],
) -> list[tuple]:
    """The (source_language, target_language, execution_engine, input_output_pairs)
    we can execute for this parallel — building each execution_engine once per
    target_language over the one shared convertible set.  Records why a parallel
    or input_output_pair is left out; returns [] (after recording why) when none
    execute.
    """
    parallel_id = parallel["parallel_id"]

    # ---- parse typed test cases from the `test` column ----
    raw_test = parallel["test"]
    if not raw_test:
        input_output_pairs_usability_report["skipped_no_test"] += 1
        return []

    parsable_input_output_pairs = test_assertion_parser.parse_test_cases(raw_test)
    if not parsable_input_output_pairs:
        input_output_pairs_usability_report["skipped_no_test_cases"] += 1
        return []

    # ---- detect node types from the Python reference signature ----
    try:
        node_types = datatypes.detect_node_types(parallel)
    except datatypes.UnsupportedInputOutputValue:
        input_output_pairs_usability_report["skipped_no_python_reference"] += 1
        return []

    # ---- filter convertible ----
    (
        convertible,
        unconvertible,
    ) = _filter_input_output_pairs_convertible_to_datatypes_in_all_supported_languages(
        parsable_input_output_pairs, node_types
    )
    input_output_pairs_usability_report["unconvertible_input_output_pair_count"] += len(
        unconvertible
    )

    if not convertible:
        input_output_pairs_usability_report[
            "skipped_no_convertible_input_output_pairs"
        ] += 1
        return []

    # ---- no expected output (PLAIN return, all None) ----
    if _has_no_expected_output(convertible, node_types):
        input_output_pairs_usability_report["skipped_no_expected_output"] += 1
        return []

    # ---- build engines per target_language over the one shared set ----
    present_languages = _present_languages(parallel)
    executable_translations: list[tuple] = []

    for target_language in present_languages:
        if (parallel_id, target_language) in exclusions:
            input_output_pairs_usability_report["excluded_by_quality_audit"] += 1
            continue
        execution_engine = _build_execution_engine(
            target_language,
            convertible,
            node_types,
            parallel["entry_point"],
            java_source=parallel.get("java") if target_language == "java" else None,
        )
        for source_language in present_languages:
            if source_language != target_language:
                executable_translations.append(
                    (
                        source_language,
                        target_language,
                        execution_engine,
                        convertible,
                    )
                )

    # ---- record kept ----
    _record_kept(
        input_output_pairs_usability_report,
        parallel_id,
        executable_translations,
        node_types,
        convertible,
    )
    return executable_translations


def _present_languages(parallel: dict) -> list[str]:
    return [lang for lang in INSTRUCT_LANGUAGES if parallel.get(lang)]


# ---- Filter convertible ----


def _filter_input_output_pairs_convertible_to_datatypes_in_all_supported_languages(
    parsable_input_output_pairs: list[dict],
    node_types: dict,
) -> tuple[list[dict], list[dict]]:
    """Split parsable pairs into those whose every value can be code in all
    of cpp/java/python and those that can't.  Rules are the same across
    languages so we check once.
    """
    convertible: list[dict] = []
    unconvertible: list[dict] = []

    for pair in parsable_input_output_pairs:
        try:
            _check_values_convertible(pair, node_types)
            convertible.append(pair)
        except datatypes.UnsupportedInputOutputValue:
            unconvertible.append(pair)

    return convertible, unconvertible


def _check_values_convertible(pair: dict, node_types: dict) -> None:
    """Verify every value in *pair* can be expressed as code in cpp/java/python.
    Raises UnsupportedInputOutputValue otherwise.
    """
    param_types = node_types.get("parameters", [])
    for i, arg in enumerate(pair["input"]):
        if i < len(param_types) and param_types[i] in (
            datatypes.LIST_NODE,
            datatypes.TREE_NODE,
        ):
            continue  # node types are convertible
        _check_plain_convertible(arg)

    ret_type = node_types.get("return_value", datatypes.PLAIN)
    if ret_type in (datatypes.LIST_NODE, datatypes.TREE_NODE):
        return  # node return types are convertible
    if pair["expected"] is None:
        raise datatypes.UnsupportedInputOutputValue(
            "expected is None with PLAIN return type — cannot compare"
        )
    _check_plain_convertible(pair["expected"])


def _check_plain_convertible(value: object) -> None:
    """Verify *value* can be expressed as a C++ literal (our strictest target)."""
    datatypes.cpp_literal(value)


def _has_no_expected_output(convertible: list[dict], node_types: dict) -> bool:
    """True if all expected outputs are None and return type is PLAIN.

    This means the problem mutates in place; there is no post-state in the
    data to compare against.  Skip the whole snippet.
    """
    if node_types.get("return_value") in (datatypes.LIST_NODE, datatypes.TREE_NODE):
        return False  # node returns have a meaningful expected value
    return all(pair.get("expected") is None for pair in convertible)


# ---- Execution engine construction ----


def _build_execution_engine(
    target_language: str,
    input_output_pairs: list[dict],
    node_types: dict,
    entry_point: str | None,
    java_source: str | None = None,
) -> str:
    """Build the target-language driver that runs a translation against its pairs.

    The engine is combined with the model's code in execution.py.
    Outputs one line per test case: "OK" or "FAIL".
    """
    entry_point = (entry_point or "solve").removeprefix("Solution().")
    if target_language == "python":
        return _python_engine(entry_point, input_output_pairs, node_types)
    elif target_language == "cpp":
        return _cpp_engine(entry_point, input_output_pairs, node_types)
    elif target_language == "java":
        return _java_engine(entry_point, input_output_pairs, node_types, java_source)
    else:
        raise datatypes.UnsupportedInputOutputValue(
            f"Unsupported target language: {target_language!r}"
        )


def _python_engine(entry_point: str, pairs: list[dict], node_types: dict) -> str:
    """Build Python execution engine, handling ListNode/TreeNode round-trip."""
    param_types = node_types.get("parameters", [])
    ret_type = node_types.get("return_value", datatypes.PLAIN)

    case_lines: list[str] = []
    for i, pair in enumerate(pairs):
        input_args: list[str] = []
        for j, arg in enumerate(pair["input"]):
            is_node = j < len(param_types) and param_types[j] in (
                datatypes.LIST_NODE,
                datatypes.TREE_NODE,
            )
            if is_node:
                if param_types[j] == datatypes.LIST_NODE:
                    input_args.append(
                        f"ListNode.from_array({datatypes.list_node_array_text(arg)})"
                    )
                else:
                    input_args.append(
                        f"TreeNode.from_array({datatypes.tree_node_array_text(arg)})"
                    )
            else:
                input_args.append(datatypes.python_literal(arg))
        call = f"_actual = _sol.{entry_point}({', '.join(input_args)})"
        expected_text = _python_expected_text(pair["expected"], ret_type, i)
        case_lines.append(
            f"try:\n    {call}\n{expected_text}\nexcept Exception:\n    print('FAIL')\n"
        )

    return "\n# === EXECUTION ENGINE ===\n_sol = Solution()\n" + "".join(case_lines)


def _python_expected_text(expected: object, ret_type: str, case_idx: int) -> str:
    """Return the body lines (4-space indent, no trailing newline) for one test case comparison."""
    if ret_type == datatypes.LIST_NODE:
        canonical = datatypes.list_node_array_text(
            expected if expected is not None else []
        )
        return f"    print('OK' if (_actual.to_array() if _actual else []) == {canonical} else 'FAIL')"
    elif ret_type == datatypes.TREE_NODE:
        canonical = datatypes.tree_node_array_text(
            expected if expected is not None else []
        )
        return f"    print('OK' if (_actual.to_array() if _actual else []) == {canonical} else 'FAIL')"
    else:
        return f"    print('OK' if _actual == {datatypes.python_literal(expected)} else 'FAIL')"


_JAVA_REPR_METHOD = (
    "    static String _repr(Object o) {\n"
    '        if (o == null) return "null";\n'
    "        if (o instanceof int[]) return java.util.Arrays.toString((int[]) o);\n"
    "        if (o instanceof long[]) return java.util.Arrays.toString((long[]) o);\n"
    "        if (o instanceof double[]) return java.util.Arrays.toString((double[]) o);\n"
    "        if (o instanceof boolean[]) return java.util.Arrays.toString((boolean[]) o);\n"
    "        if (o instanceof char[]) return java.util.Arrays.toString((char[]) o);\n"
    "        if (o instanceof Object[]) {\n"
    "            Object[] a = (Object[]) o;\n"
    '            StringBuilder sb = new StringBuilder("[");\n'
    '            for (int k = 0; k < a.length; k++) { if (k > 0) sb.append(", "); sb.append(_repr(a[k])); }\n'
    '            return sb.append("]").toString();\n'
    "        }\n"
    "        if (o instanceof java.util.List) {\n"
    "            java.util.List<?> a = (java.util.List<?>) o;\n"
    '            StringBuilder sb = new StringBuilder("[");\n'
    '            for (int k = 0; k < a.size(); k++) { if (k > 0) sb.append(", "); sb.append(_repr(a.get(k))); }\n'
    '            return sb.append("]").toString();\n'
    "        }\n"
    "        return String.valueOf(o);\n"
    "    }\n"
)


def _wide_arg_positions(pairs: list[dict]) -> list[bool]:
    """For each argument position, True if any pair's value there needs 64-bit ints."""
    n = max((len(p["input"]) for p in pairs), default=0)
    return [
        any(datatypes.needs_int64(p["input"][j]) for p in pairs if j < len(p["input"]))
        for j in range(n)
    ]


def _java_cast(java_type_str: str) -> str:
    """Cast expression fragment for unboxing from Object[]; primitives need a double cast."""
    return {
        "int": "int)(Integer",
        "long": "long)(Long",
        "double": "double)(Double",
        "float": "float)(Float",
        "boolean": "boolean)(Boolean",
        "char": "char)(Character",
    }.get(java_type_str.strip(), java_type_str)


def _cpp_engine(entry_point: str, pairs: list[dict], node_types: dict) -> str:
    param_types = node_types.get("parameters", [])
    param_hints = node_types.get("parameter_hints", [])
    ret_type = node_types.get("return_value", datatypes.PLAIN)
    wide_args = _wide_arg_positions(pairs)
    n_cases = len(pairs)
    n_args = max((len(p["input"]) for p in pairs), default=0)

    decls: list[str] = []
    args_i: list[str] = []
    args_0: list[str] = []
    for j in range(n_args):
        column = [p["input"][j] for p in pairs]
        node = param_types[j] if j < len(param_types) else datatypes.PLAIN
        if node == datatypes.LIST_NODE:
            lits = ", ".join(datatypes.cpp_literal(v) for v in column)
            decls.append(f"    std::vector<std::vector<int>> _data{j} = {{{lits}}};")
            args_i.append(f"_arrayToListNode(_data{j}[i])")
            args_0.append(f"_arrayToListNode(_data{j}[0])")
        elif node == datatypes.TREE_NODE:
            lits = ", ".join(
                datatypes.cpp_optional_int_vector_literal(v) for v in column
            )
            decls.append(
                f"    std::vector<std::vector<std::optional<int>>> _data{j} = {{{lits}}};"
            )
            args_i.append(f"_arrayToTreeNode(_data{j}[i])")
            args_0.append(f"_arrayToTreeNode(_data{j}[0])")
        else:
            hint = param_hints[j] if j < len(param_hints) else ""
            wide = wide_args[j] if j < len(wide_args) else False
            elem_t = datatypes.cpp_type_from_hint(
                hint, column[0] if column else None, wide=wide
            )
            lits = ", ".join(datatypes.cpp_literal(v) for v in column)
            decls.append(f"    std::vector<{elem_t}> _data{j} = {{{lits}}};")
            args_i.append(f"_data{j}[i]")
            args_0.append(f"_data{j}[0]")

    call_i = f"s.{entry_point}({', '.join(args_i)})"
    call_0 = f"s.{entry_point}({', '.join(args_0)})"

    if ret_type == datatypes.LIST_NODE:
        exp = ", ".join(
            datatypes.cpp_literal(
                p["expected"] if isinstance(p["expected"], list) else []
            )
            for p in pairs
        )
        expected_decl = f"    std::vector<std::vector<int>> _expected = {{{exp}}};"
        body = (
            f"            auto _result = {call_i};\n"
            "            auto _result_arr = _listNodeToArray(_result);\n"
            '            std::cout << (_result_arr == _expected[i] ? "OK" : "FAIL") << "\\n";'
        )
    elif ret_type == datatypes.TREE_NODE:
        exp = ", ".join(
            datatypes.cpp_optional_int_vector_literal(
                p["expected"] if isinstance(p["expected"], list) else []
            )
            for p in pairs
        )
        expected_decl = (
            f"    std::vector<std::vector<std::optional<int>>> _expected = {{{exp}}};"
        )
        body = (
            f"            auto _result = {call_i};\n"
            "            auto _result_arr = _treeNodeToArray(_result);\n"
            '            std::cout << (_result_arr == _expected[i] ? "OK" : "FAIL") << "\\n";'
        )
    else:
        exp = ", ".join(datatypes.cpp_literal(p["expected"]) for p in pairs)
        expected_decl = (
            f"    using _RetT = decltype({call_0});\n"
            f"    std::vector<_RetT> _expected = {{{exp}}};"
        )
        body = (
            f"            auto _result = {call_i};\n"
            '            std::cout << (_result == _expected[i] ? "OK" : "FAIL") << "\\n";'
        )

    return (
        "\n// === EXECUTION ENGINE ===\n"
        "int main() {\n"
        "    Solution s;\n"
        + "\n".join(decls)
        + ("\n" if decls else "")
        + expected_decl
        + "\n"
        + f"    for (size_t i = 0; i < {n_cases}; i++) {{\n"
        + "        try {\n"
        + body
        + "\n"
        + '        } catch (...) { std::cout << "FAIL" << "\\n"; }\n'
        + "    }\n    return 0;\n}\n"
    )


def _java_engine(
    entry_point: str,
    pairs: list[dict],
    node_types: dict,
    java_source: str | None = None,
) -> str:
    param_types = node_types.get("parameters", [])
    ret_type = node_types.get("return_value", datatypes.PLAIN)
    java_sig_types = (
        datatypes.parse_java_param_types(java_source, entry_point)
        if java_source
        else None
    )
    wide_args = _wide_arg_positions(pairs)
    n_cases = len(pairs)
    n_args = max((len(p["input"]) for p in pairs), default=0)

    decls: list[str] = []
    call_args: list[str] = []
    for j in range(n_args):
        column = [p["input"][j] for p in pairs]
        node = param_types[j] if j < len(param_types) else datatypes.PLAIN
        if node == datatypes.LIST_NODE:
            lits = ", ".join(datatypes.java_literal(v) for v in column)
            decls.append(f"        int[][] _data{j} = {{{lits}}};")
            call_args.append(f"_NodeHelpers.arrayToListNode(_data{j}[i])")
        elif node == datatypes.TREE_NODE:
            lits = ", ".join(
                datatypes.java_nullable_int_array_literal(v) for v in column
            )
            decls.append(f"        Integer[][] _data{j} = {{{lits}}};")
            call_args.append(f"_NodeHelpers.arrayToTreeNode(_data{j}[i])")
        else:
            jt = (
                java_sig_types[j]
                if (java_sig_types and j < len(java_sig_types))
                else (datatypes.java_type(column[0]) if column else "Object")
            )
            if j < len(wide_args) and wide_args[j]:
                jt = datatypes.widen_java_int_type(jt)
            lits = ", ".join(datatypes.java_literal_for_type(v, jt) for v in column)
            decls.append(f"        Object[] _data{j} = {{{lits}}};")
            call_args.append(f"({_java_cast(jt)}) _data{j}[i]")

    call = f"_s.{entry_point}({', '.join(call_args)})"

    if ret_type == datatypes.LIST_NODE:
        exp = ", ".join(
            datatypes.java_literal(
                p["expected"] if isinstance(p["expected"], list) else []
            )
            for p in pairs
        )
        expected_decl = f"        int[][] _expected = {{{exp}}};"
        body = (
            f"                ListNode _result = {call};\n"
            "                int[] _result_arr = _NodeHelpers.listNodeToArray(_result);\n"
            '                System.out.println(java.util.Arrays.equals(_result_arr, _expected[i]) ? "OK" : "FAIL");'
        )
    elif ret_type == datatypes.TREE_NODE:
        exp = ", ".join(
            datatypes.java_nullable_int_array_literal(
                p["expected"] if isinstance(p["expected"], list) else []
            )
            for p in pairs
        )
        expected_decl = f"        Integer[][] _expected = {{{exp}}};"
        body = (
            f"                TreeNode _result = {call};\n"
            "                Integer[] _result_arr = _NodeHelpers.treeNodeToArray(_result);\n"
            '                System.out.println(java.util.Arrays.equals(_result_arr, _expected[i]) ? "OK" : "FAIL");'
        )
    else:
        exp = ", ".join(datatypes.java_literal(p["expected"]) for p in pairs)
        expected_decl = f"        Object[] _expected = {{{exp}}};"
        body = (
            f"                Object _result = {call};\n"
            '                System.out.println(_repr(_result).equals(_repr(_expected[i])) ? "OK" : "FAIL");'
        )

    main = (
        "    public static void main(String[] _args) {\n"
        "        Solution _s = new Solution();\n"
        + "\n".join(decls)
        + ("\n" if decls else "")
        + expected_decl
        + "\n"
        + f"        for (int i = 0; i < {n_cases}; i++) {{\n"
        + "            try {\n"
        + body
        + "\n"
        + '            } catch (Throwable _e) { System.out.println("FAIL"); }\n'
        + "        }\n    }\n"
    )
    return (
        "\n// === EXECUTION ENGINE ===\nclass Main {\n"
        + _JAVA_REPR_METHOD
        + "\n"
        + main
        + "}\n"
    )


# ---- Usability report ----


def _new_report() -> dict:
    """A plain dict (not a class) recording which snippets/pairs were usable and why the rest were skipped."""
    return {
        "kept": 0,
        "kept_with_node_types": 0,
        "kept_input_output_pair_count": 0,
        "kept_parallel_ids_by_target_language": collections.defaultdict(set),
        "kept_payloads_by_target_language": collections.Counter(),
        "skipped_no_test": 0,
        "skipped_no_test_cases": 0,
        "skipped_no_python_reference": 0,
        "skipped_no_expected_output": 0,
        "skipped_no_convertible_input_output_pairs": 0,
        "excluded_by_quality_audit": 0,
        "unconvertible_input_output_pair_count": 0,
    }


def _record_kept(
    report: dict,
    parallel_id: int,
    executable_translations: list[tuple],
    node_types: dict,
    convertible_pairs: list[dict],
) -> None:
    report["kept"] += len(executable_translations)
    report["kept_input_output_pair_count"] += len(convertible_pairs)
    has_nodes = any(
        t in (datatypes.LIST_NODE, datatypes.TREE_NODE)
        for t in node_types.get("parameters", []) + [node_types.get("return_value")]
    )
    if has_nodes:
        report["kept_with_node_types"] += len(executable_translations)
    for _, target_language, _, _ in executable_translations:
        report["kept_parallel_ids_by_target_language"][target_language].add(parallel_id)
        report["kept_payloads_by_target_language"][target_language] += 1


def _print_report(report: dict) -> None:
    ids_by_lang = report["kept_parallel_ids_by_target_language"]
    kept_parallels = len(set().union(*ids_by_lang.values())) if ids_by_lang else 0
    kept_payloads = report["kept"]
    kept_pairs = report["kept_input_output_pair_count"]
    avg_pairs = kept_pairs / kept_parallels if kept_parallels else 0

    skipped = (
        report["skipped_no_test"]
        + report["skipped_no_test_cases"]
        + report["skipped_no_python_reference"]
        + report["skipped_no_expected_output"]
        + report["skipped_no_convertible_input_output_pairs"]
    )
    total_parallels = kept_parallels + skipped

    payloads_by_lang = report["kept_payloads_by_target_language"]

    def _parallels(n: int) -> str:
        return f"{n:,} parallel{'s' if n != 1 else ''}"

    def _payloads(n: int) -> str:
        return f"{n:,} bigcode task payload{'s' if n != 1 else ''}"

    def _cases(n: int) -> str:
        return f"{n:,} test case{'s' if n != 1 else ''}"

    print(f"\nEvaluation dataset: {OUTPUT_PATH}")

    print(
        f"\n  {_parallels(kept_parallels)} / {total_parallels:,} held-out ({kept_parallels / total_parallels * 100:.0f}%)"
        f"  →  {_payloads(kept_payloads)}  ({_cases(kept_pairs)}, avg {avg_pairs:.1f}/parallel)"
    )
    if report["kept_with_node_types"]:
        node_n = report["kept_with_node_types"]
        print(f"     {_payloads(node_n)} involve ListNode / TreeNode")

    print("\n  Per target language (all share the same test cases per parallel):")
    for lang in INSTRUCT_LANGUAGES:
        n_parallels = len(ids_by_lang.get(lang, set()))
        n_payloads = payloads_by_lang.get(lang, 0)
        print(f"    {lang:6s}  {_parallels(n_parallels)}  →  {_payloads(n_payloads)}")

    print(f"\n  Skipped: {_parallels(skipped)}")
    for label_fn, key in [
        (
            lambda n: f"{_parallels(n)} absent from newfacade source (no test column)",
            "skipped_no_test",
        ),
        (
            lambda n: f"{_parallels(n)} with no parseable test cases",
            "skipped_no_test_cases",
        ),
        (
            lambda n: (
                f"{_parallels(n)} with no Python code snippet (required for ListNode / TreeNode detection)"
            ),
            "skipped_no_python_reference",
        ),
        (
            lambda n: (
                f"{_parallels(n)} where all input output pairs contain values not expressible in target languages"
            ),
            "skipped_no_convertible_input_output_pairs",
        ),
    ]:
        v = report[key]
        if v:
            print(f"    {v:>4}  {label_fn(v)}")

    if report["excluded_by_quality_audit"]:
        v = report["excluded_by_quality_audit"]
        print(f"    {v:>4}  {_payloads(v)} excluded by quality audit")
    if report["unconvertible_input_output_pair_count"]:
        v = report["unconvertible_input_output_pair_count"]
        print(f"\n  {_cases(v)} parsed but not expressible in target language code")


if __name__ == "__main__":
    main()
