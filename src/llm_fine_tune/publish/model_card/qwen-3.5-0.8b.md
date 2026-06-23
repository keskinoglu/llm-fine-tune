---
base_model: Qwen/Qwen3.5-0.8B
license: apache-2.0
datasets:
  - tkeskin/leetcode-solutions
language:
  - en
pipeline_tag: text-generation
library_name: transformers
tags:
  - lora
  - llama-factory
  - code
  - code-translation
  - qwen
  - qwen3
---

# qwen-3.5-0.8b-code-translation

A fine-tuned version of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) for translating code between **C++**, **Java**, and **Python**.

## Training

- **Base model:** Qwen/Qwen3.5-0.8B
- **Method:** LoRA (Low-Rank Adaptation) via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Dataset:** [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) (`instruct` config) — directed C++/Java/Python translation pairs derived from LeetCode solutions
- **Hardware:** AMD MI210 (ROCm) / NVIDIA CUDA, `flash_attn: sdpa`
- **Template:** `qwen3_5_nothink` (non-thinking mode — direct code output, no chain-of-thought)
- **LoRA target:** all linear layers (`lora_target: all`)
- **Precision:** bf16

## Evaluation

Evaluated with the same **execution-based** translation benchmark as the rest of this series: each held-out `evaluation`-config payload from [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) is compiled and run against its input/output pairs. Held out from training (no leakage). Metric is **pass@1**, n-weighted over 3,336 payloads.

**This fine-tune did not improve translation performance over the base model** — pass@1 is essentially unchanged, fractionally lower (within noise):

| | Base (Qwen3.5-0.8B) | This model | Δ |
|---|---|---|---|
| **pass@1** | 16.2% | 15.7% | −0.5 |
| **compile rate** | 49.5% | 49.9% | +0.4 |

For contrast, the *same* LoRA recipe and dataset lifted a larger code-pretrained base (Qwen2.5-Coder-1.5B) from 29.3% to 61.9% pass@1. The benefit appears to depend on model scale and code pre-training; at 0.8B, on this unusual architecture, the recipe did not transfer. We publish this result as-is rather than omit it.

pass@1 by language pair × difficulty (%):

| source | target | difficulty | base | this model |
|---|---|---|---|---|
| cpp | java | Easy | 11.7 | 13.8 |
| cpp | java | Hard | 2.5 | 2.5 |
| cpp | java | Medium | 10.1 | 8.5 |
| cpp | python | Easy | 22.1 | 18.0 |
| cpp | python | Hard | 11.5 | 9.2 |
| cpp | python | Medium | 18.8 | 18.5 |
| java | cpp | Easy | 14.3 | 17.7 |
| java | cpp | Hard | 1.7 | 4.2 |
| java | cpp | Medium | 12.2 | 10.0 |
| java | python | Easy | 36.6 | 34.9 |
| java | python | Hard | 16.0 | 10.7 |
| java | python | Medium | 27.9 | 27.3 |
| python | cpp | Easy | 40.1 | 38.1 |
| python | cpp | Hard | 7.6 | 10.1 |
| python | cpp | Medium | 20.7 | 24.1 |
| python | java | Easy | 13.1 | 11.0 |
| python | java | Hard | 0.8 | 0.8 |
| python | java | Medium | 5.4 | 4.3 |

Full methodology is in the [llm-fine-tune](https://github.com/tkeskin/llm-fine-tune) repo (Stage 5).

## Intended use

Given source code in one of C++, Java, or Python, the model generates a translation into the target language, following the same logic and structure.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tkeskin/qwen-3.5-0.8b-code-translation"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": "Translate the following C++ code to Python:\n\nint add(int a, int b) { return a + b; }"
    }
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```
