"""Phase 3 (code-benchmark track): generate MultiPL-E completions with an instruct-wrapped prompt.

Loads each humaneval-{cpp,java,py} config from nuprl/MultiPL-E, wraps each prompt as a chat
instruction, generates via the model, extracts code, and writes code_benchmark_generations.parquet
plus a raw JSON debug file — both from the same rows in order so Phase-4 can pair by index.
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
_LANG_MAP = {"cpp": "cpp", "java": "java", "py": "python"}


def _render_prompt(tokenizer, lang_suffix: str, prompt_text: str) -> str:
    content = (
        f"Complete this {lang_suffix} function. "
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
        lang_suffix = config.split("-")[-1]  # "humaneval-cpp" → "cpp"
        language = _LANG_MAP[lang_suffix]

        rows = list(load_dataset(_MULTIPL_E_REPO, config)["test"])
        if args.limit:
            rows = rows[: args.limit]

        prompts = [_render_prompt(tokenizer, lang_suffix, r["prompt"]) for r in rows]
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
                raw_completion, language
            )
            all_rows.append(
                {
                    "config": config,
                    "name": row["name"],
                    "language": language,
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
