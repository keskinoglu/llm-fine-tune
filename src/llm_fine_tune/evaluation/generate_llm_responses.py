"""Phase 1: generate an llm_response for each bigcode_task_payload with a (fine-tuned) model.

Loads the evaluation dataset, asks the model to translate each code_snippet_to_translate, and
writes generations.json plus the evaluation parquet — both from the same rows in one pass, so
the Phase-2 scorer's row-pairing can't drift. Plain transformers; generation needs no harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fine_tune.evaluation.generation import generate_completions

_DATASET_PATH = "tkeskin/leetcode-solutions"
_DATASET_CONFIG = "evaluation"


def _build_prompt(payload: dict) -> str:
    return f"{payload['user_prompt']}\n\n{payload['code_snippet_to_translate']}"


def _render_prompt(tokenizer, payload: dict) -> str:
    prompt = _build_prompt(payload)
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def main() -> None:
    args = _parse_args()

    payloads = list(load_dataset(_DATASET_PATH, _DATASET_CONFIG)["test"])
    if args.limit:
        payloads = payloads[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Decoder-only batching: pad on the left so the generated tokens are a clean suffix
    # at the same offset for every row in the batch.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(
        "cuda"
    )
    model.eval()

    prompts = [_render_prompt(tokenizer, p) for p in payloads]
    raw = generate_completions(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature or 0.0,
        batch_size=args.batch_size,
    )
    generations = [[r] for r in raw]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generations.json").write_text(json.dumps(generations, indent=2))
    pl.from_dicts(payloads).write_parquet(out_dir / "evaluation.parquet")
    print(f"Generated {len(generations)} responses -> {out_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate translation llm_responses for the evaluation dataset."
    )
    parser.add_argument("--model", required=True, help="HF id or local path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Writes generations.json + evaluation.parquet here (Phase-2 reads both).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N payloads (shakeout)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
