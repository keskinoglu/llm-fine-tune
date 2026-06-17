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

_DATASET_PATH = "tkeskin/leetcode-solutions"
_DATASET_CONFIG = "evaluation"


def _build_prompt(payload: dict) -> str:
    return f"{payload['user_prompt']}\n\n{payload['code_snippet_to_translate']}"


def _encode(tokenizer, prompt: str):
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    return tokenizer(prompt, return_tensors="pt").input_ids


def main() -> None:
    args = _parse_args()

    payloads = list(load_dataset(_DATASET_PATH, _DATASET_CONFIG)["test"])
    if args.limit:
        payloads = payloads[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.temperature and args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)
    else:
        gen_kwargs.update(do_sample=False)

    generations = []
    for payload in payloads:
        input_ids = _encode(tokenizer, _build_prompt(payload)).to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids, **gen_kwargs)
        llm_response = tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        generations.append([llm_response])

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
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N payloads (shakeout)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
