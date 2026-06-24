"""Phase 3 (code-benchmark track): generate HumanEval/MultiPL-E completions with an instruct prompt.

The non-Python languages come from nuprl/MultiPL-E (HumanEval translated out of Python), which
ships a self-checking test harness per row. Python is the source language of MultiPL-E and so has
no config there; it comes from the canonical openai/openai_humaneval, whose `test` defines a
`check(candidate)` function that we close over the entry point to make self-checking too.

Each config is normalized to the same row shape — name, language, prompt, tests, completion — so
Phase-4 scoring (assemble preamble + completion + tests, run, pass = exit 0) is source-agnostic.
We wrap each prompt as a chat instruction, generate, extract the code, and write
code_benchmark_generations.parquet plus a raw JSON debug file in row order so Phase-4 pairs by index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fine_tune.evaluation import extract_code_snippet_from_llm_response as extractor
from llm_fine_tune.evaluation.generation import generate_completions

_MULTIPL_E_REPO = "nuprl/MultiPL-E"
_HUMANEVAL_REPO = "openai/openai_humaneval"

# MultiPL-E config suffix -> (our execution language token, display name for the instruction).
# Python is not in MultiPL-E (it is the source language); it is handled separately below.
_MULTIPL_E_LANGS = {
    "cpp": ("cpp", "C++"),
    "java": ("java", "Java"),
    "rs": ("rust", "Rust"),
    "go": ("go", "Go"),
}
_PYTHON_CONFIGS = {"humaneval-py", "humaneval-python"}


def _load_normalized_rows(config: str, limit: int | None) -> list[dict]:
    """Return rows shaped {name, language, prompt, tests} for any supported config.

    `tests` is a self-checking harness in the row's language: appending the model's completion
    and running it exits 0 iff every assertion passes.
    """
    if config in _PYTHON_CONFIGS:
        rows = list(load_dataset(_HUMANEVAL_REPO)["test"])
        if limit:
            rows = rows[:limit]
        return [
            {
                "name": r["task_id"],
                "language": "python",
                "display": "Python",
                "prompt": r["prompt"],
                # `test` defines check(candidate); close it over the function name to self-check.
                "tests": r["test"] + f"\n\ncheck({r['entry_point']})\n",
            }
            for r in rows
        ]

    suffix = config.split("-")[-1]  # "humaneval-cpp" → "cpp"
    if suffix not in _MULTIPL_E_LANGS:
        raise ValueError(
            f"Unsupported config {config!r}; supported suffixes: "
            f"{sorted(_MULTIPL_E_LANGS)} plus python via {sorted(_PYTHON_CONFIGS)}"
        )
    language, display = _MULTIPL_E_LANGS[suffix]
    rows = list(load_dataset(_MULTIPL_E_REPO, config)["test"])
    if limit:
        rows = rows[:limit]
    return [
        {
            "name": r["name"],
            "language": language,
            "display": display,
            "prompt": r["prompt"],
            "tests": r["tests"],
        }
        for r in rows
    ]


def _render_prompt(tokenizer, display: str, prompt_text: str) -> str:
    content = (
        f"Complete this {display} function. "
        f"Respond with only the complete function in a single code block.\n\n{prompt_text}"
    )
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return content


def main() -> None:
    args = _parse_args()
    configs = [c.strip() for c in args.configs.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(
        "cuda"
    )
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    raw_by_config: dict[str, list[str]] = {}

    for config in configs:
        rows = _load_normalized_rows(config, args.limit)

        prompts = [_render_prompt(tokenizer, r["display"], r["prompt"]) for r in rows]
        completions = generate_completions(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

        raw_by_config[config] = completions
        for row, raw_completion in zip(rows, completions):
            extracted = extractor.extract_code_snippet_from_llm_response(
                raw_completion, row["language"]
            )
            all_rows.append(
                {
                    "config": config,
                    "name": row["name"],
                    "language": row["language"],
                    "prompt": row["prompt"],
                    "tests": row["tests"],
                    "completion": extracted,
                }
            )

    pl.from_dicts(all_rows).write_parquet(
        out_dir / "code_benchmark_generations.parquet"
    )
    (out_dir / "code_benchmark_generations.json").write_text(
        json.dumps(raw_by_config, indent=2)
    )
    print(
        f"Generated {len(all_rows)} completions -> {out_dir}/code_benchmark_generations.parquet"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MultiPL-E completions (Phase 3, code-benchmark track)."
    )
    parser.add_argument("--model", required=True, help="HF id or local path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--configs",
        default="humaneval-cpp,humaneval-java,humaneval-py",
        help="Comma-separated nuprl/MultiPL-E config names.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N rows per config (shakeout).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
