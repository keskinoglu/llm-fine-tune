"""Shared batched generation loop for evaluation scripts."""

from __future__ import annotations

import torch


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> list[str]:
    """Run batched left-padded generation; return one decoded string per prompt."""
    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    completions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        inputs = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        for sequence in output[:, inputs["input_ids"].shape[1] :]:
            completions.append(tokenizer.decode(sequence, skip_special_tokens=True))
    return completions
