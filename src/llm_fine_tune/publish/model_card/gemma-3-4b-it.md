---
base_model: google/gemma-3-4b-it
license: gemma
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
  - gemma
  - gemma-3
---

# gemma-3-4b-it-code-translation

A fine-tuned version of [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) for translating code between **C++**, **Java**, and **Python**.

## Training

- **Base model:** google/gemma-3-4b-it
- **Method:** LoRA (Low-Rank Adaptation) via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Dataset:** [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) (`instruct` config) — directed C++/Java/Python translation pairs derived from LeetCode solutions
- **Hardware:** AMD MI210 (ROCm) / NVIDIA CUDA, `flash_attn: sdpa`
- **LoRA target:** all linear layers (`lora_target: all`)
- **Precision:** bf16

## Evaluation

Evaluated with an **execution-based** translation benchmark: each held-out `evaluation`-config payload from [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) is a directed source→target translation whose output is compiled and run against the problem's input/output pairs. The eval split is held out from training (no leakage). Metric is **pass@1** (all test cases pass), n-weighted over 3,336 payloads.

| | Base (gemma-3-4b-it) | This model | Δ |
|---|---|---|---|
| **pass@1** | 27.9% | **52.9%** | **+25.0** |
| **compile rate** | 54.9% | **79.5%** | **+24.6** |

pass@1 by language pair × difficulty (%):

| source | target | difficulty | base | this model |
|---|---|---|---|---|
| cpp | java | Easy | 33.1 | 68.3 |
| cpp | java | Hard | 12.7 | 34.7 |
| cpp | java | Medium | 20.5 | 59.7 |
| cpp | python | Easy | 54.7 | 66.3 |
| cpp | python | Hard | 32.8 | 38.9 |
| cpp | python | Medium | 39.0 | 57.8 |
| java | cpp | Easy | 63.9 | 80.3 |
| java | cpp | Hard | 23.5 | 44.5 |
| java | cpp | Medium | 43.3 | 68.5 |
| java | python | Easy | 36.0 | 67.4 |
| java | python | Hard | 21.4 | 41.2 |
| java | python | Medium | 27.9 | 57.1 |
| python | cpp | Easy | 20.4 | 57.8 |
| python | cpp | Hard | 4.2 | 20.2 |
| python | cpp | Medium | 11.1 | 40.7 |
| python | java | Easy | 24.1 | 54.5 |
| python | java | Hard | 5.9 | 18.6 |
| python | java | Medium | 14.3 | 41.1 |

Full methodology is in the [llm-fine-tune](https://github.com/tkeskin/llm-fine-tune) repo (Stage 5).

## Intended use

Given source code in one of C++, Java, or Python, the model generates a translation into the target language, following the same logic and structure.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tkeskin/gemma-3-4b-it-code-translation"
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
