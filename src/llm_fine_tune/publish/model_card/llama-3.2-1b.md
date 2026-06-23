---
base_model: meta-llama/Llama-3.2-1B-Instruct
license: llama3.2
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
  - llama
---

# llama-3.2-1b-instruct-code-translation

A fine-tuned version of [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) for translating code between **C++**, **Java**, and **Python**.

## Training

- **Base model:** meta-llama/Llama-3.2-1B-Instruct
- **Method:** LoRA (Low-Rank Adaptation) via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Dataset:** [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) (`instruct` config) — directed C++/Java/Python translation pairs derived from LeetCode solutions
- **Hardware:** AMD MI210 (ROCm) / NVIDIA CUDA, `flash_attn: sdpa`
- **LoRA target:** all linear layers (`lora_target: all`)
- **Precision:** bf16

## Evaluation

Evaluated with an **execution-based** translation benchmark: each held-out `evaluation`-config payload from [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) is a directed source→target translation whose output is compiled and run against the problem's input/output pairs. The eval split is held out from training (no leakage). Metric is **pass@1** (all test cases pass), n-weighted over 3,336 payloads.

| | Base (Llama-3.2-1B-Instruct) | This model | Δ |
|---|---|---|---|
| **pass@1** | 17.5% | **32.5%** | **+15.0** |
| **compile rate** | 52.8% | **72.7%** | **+19.8** |

pass@1 by language pair × difficulty (%):

| source | target | difficulty | base | this model |
|---|---|---|---|---|
| cpp | java | Easy | 29.0 | 55.9 |
| cpp | java | Hard | 7.6 | 24.6 |
| cpp | java | Medium | 15.9 | 39.5 |
| cpp | python | Easy | 32.0 | 37.2 |
| cpp | python | Hard | 10.7 | 17.6 |
| cpp | python | Medium | 24.4 | 33.8 |
| java | cpp | Easy | 15.0 | 61.2 |
| java | cpp | Hard | 4.2 | 27.7 |
| java | cpp | Medium | 14.4 | 44.4 |
| java | python | Easy | 31.4 | 44.8 |
| java | python | Hard | 13.0 | 22.9 |
| java | python | Medium | 19.2 | 31.8 |
| python | cpp | Easy | 23.8 | 40.1 |
| python | cpp | Hard | 1.7 | 6.7 |
| python | cpp | Medium | 15.6 | 23.3 |
| python | java | Easy | 24.1 | 30.3 |
| python | java | Hard | 3.4 | 7.6 |
| python | java | Medium | 12.4 | 19.4 |

Full methodology is in the [llm-fine-tune](https://github.com/tkeskin/llm-fine-tune) repo (Stage 5).

## Intended use

Given source code in one of C++, Java, or Python, the model generates a translation into the target language, following the same logic and structure.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tkeskin/llama-3.2-1b-instruct-code-translation"
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
