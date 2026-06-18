"""Compute held-out perplexity of a model on the leetcode_instruct_test split.

Perplexity is measured on the *completion* tokens only (the expected code translation),
conditioned on the instruction+input prompt. Prompt tokens are masked from the loss.

Writes perplexity.json: {"model": str, "perplexity": float, "n_samples": int, "n_tokens": int}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_DATASET_PATH = "tkeskin/leetcode-solutions"
_DATASET_CONFIG = "instruct"


def _build_messages(row: dict) -> tuple[list[dict], str]:
    user_content = row["instruction"]
    if row.get("input"):
        user_content = f"{row['instruction']}\n\n{row['input']}"
    return [{"role": "user", "content": user_content}], row["output"]


def _compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[dict],
    batch_size: int,
) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]

        prompt_ids_list = []
        full_ids_list = []
        for row in batch:
            prompt_messages, completion = _build_messages(row)

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            full_ids = tokenizer.apply_chat_template(
                prompt_messages + [{"role": "assistant", "content": completion}],
                add_generation_prompt=False,
                tokenize=True,
            )
            prompt_ids_list.append(prompt_ids)
            full_ids_list.append(full_ids)

        max_len = max(len(ids) for ids in full_ids_list)
        input_ids = torch.full(
            (len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, (prompt_ids, full_ids) in enumerate(zip(prompt_ids_list, full_ids_list)):
            seq_len = len(full_ids)
            prompt_len = len(prompt_ids)
            input_ids[i, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
            # Only supervise on completion tokens (shift: model predicts next token)
            comp_start = prompt_len  # first completion token position
            labels[i, comp_start:seq_len] = torch.tensor(
                full_ids[comp_start:], dtype=torch.long
            )

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)

        # out.loss is mean NLL over non-masked tokens across the batch
        n_completion_tokens = (labels != -100).sum().item()
        total_nll += out.loss.item() * n_completion_tokens
        total_tokens += n_completion_tokens

    return total_nll, total_tokens


def main() -> None:
    args = _parse_args()

    rows = list(load_dataset(_DATASET_PATH, _DATASET_CONFIG)["test"])
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(
        "cuda"
    )
    model.eval()

    total_nll, total_tokens = _compute_perplexity(
        model, tokenizer, rows, args.batch_size
    )
    perplexity = math.exp(total_nll / total_tokens)

    result = {
        "model": args.model,
        "perplexity": perplexity,
        "n_samples": len(rows),
        "n_tokens": total_tokens,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "perplexity.json").write_text(json.dumps(result, indent=2))
    print(
        f"Perplexity: {perplexity:.2f}  ({total_tokens:,} completion tokens, {len(rows)} samples)"
    )
    print(f"Written -> {out_dir / 'perplexity.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute held-out perplexity on leetcode_instruct_test."
    )
    parser.add_argument("--model", required=True, help="HF id or local path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Writes perplexity.json here.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N samples (shakeout)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
