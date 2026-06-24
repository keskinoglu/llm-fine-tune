---
base_model: mistralai/Mistral-7B-Instruct-v0.3
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
  - mistral
---

# mistral-7b-v0.3-code-translation

A fine-tuned version of [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) for translating code between **C++**, **Java**, and **Python**.

## Training

- **Base model:** mistralai/Mistral-7B-Instruct-v0.3
- **Method:** LoRA (Low-Rank Adaptation) via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Dataset:** [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) (`instruct` config) — directed C++/Java/Python translation pairs derived from LeetCode solutions
- **Hardware:** AMD MI210 (ROCm) / NVIDIA CUDA, `flash_attn: sdpa`
- **LoRA target:** all linear layers (`lora_target: all`)
- **Precision:** bf16

## Evaluation

Evaluated with an **execution-based** translation benchmark: each held-out `evaluation`-config payload from [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) is a directed source→target translation whose output is compiled and run against the problem's input/output pairs. The eval split is held out from training (no leakage). Metric is **pass@1** (all test cases pass), n-weighted over 3,336 payloads.

| | Base (Mistral-7B-Instruct-v0.3) | This model | Δ |
|---|---|---|---|
| **pass@1** | 11.8% | **59.6%** | **+47.8** |
| **compile rate** | 45.0% | **84.7%** | **+39.6** |

This is the **largest gain in the series** — the base model barely produces compilable C++/Java, and fine-tuning lifts it to near the top. pass@1 by language pair × difficulty (%):

| source | target | difficulty | base | this model |
|---|---|---|---|---|
| cpp | java | Easy | 9.7 | 81.4 |
| cpp | java | Hard | 3.4 | 47.5 |
| cpp | java | Medium | 8.1 | 72.9 |
| cpp | python | Easy | 29.7 | 61.6 |
| cpp | python | Hard | 16.0 | 45.8 |
| cpp | python | Medium | 27.3 | 64.3 |
| java | cpp | Easy | 2.7 | 79.6 |
| java | cpp | Hard | 2.5 | 51.3 |
| java | cpp | Medium | 4.1 | 70.0 |
| java | python | Easy | 4.1 | 64.5 |
| java | python | Hard | 6.1 | 45.8 |
| java | python | Medium | 8.4 | 62.3 |
| python | cpp | Easy | 19.7 | 64.6 |
| python | cpp | Hard | 10.1 | 26.9 |
| python | cpp | Medium | 17.8 | 51.9 |
| python | java | Easy | 17.2 | 64.8 |
| python | java | Hard | 2.5 | 28.8 |
| python | java | Medium | 8.5 | 53.1 |

Full methodology is in the [llm-fine-tune](https://github.com/tkeskin/llm-fine-tune) repo (Stage 5).

## Intended use

Given source code in one of C++, Java, or Python, the model generates a translation into the target language, following the same logic and structure.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tkeskin/mistral-7b-v0.3-code-translation"
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
