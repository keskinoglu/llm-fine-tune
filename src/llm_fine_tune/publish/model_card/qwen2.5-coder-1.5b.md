---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
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
  - qwen2
  - code
---

# qwen2.5-coder-1.5b-code-translation

A fine-tuned version of [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) for translating code between **C++**, **Java**, and **Python**.

## Training

- **Base model:** Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Method:** LoRA (Low-Rank Adaptation) via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Dataset:** [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) (`instruct` config) — directed C++/Java/Python translation pairs derived from LeetCode solutions
- **Hardware:** AMD MI210 (ROCm) / NVIDIA CUDA, `flash_attn: sdpa`
- **LoRA target:** all linear layers (`lora_target: all`)
- **Precision:** bf16

## Evaluation

Evaluated with an **execution-based** translation benchmark: each held-out `evaluation`-config payload from [tkeskin/leetcode-solutions](https://huggingface.co/datasets/tkeskin/leetcode-solutions) is a directed source→target translation whose output is compiled and run against the problem's input/output pairs. The eval split is held out from training (no leakage). Metric is **pass@1** (all test cases pass), n-weighted over 3,336 payloads.

| | Base (Qwen2.5-Coder-1.5B-Instruct) | This model | Δ |
|---|---|---|---|
| **pass@1** | 29.3% | **61.9%** | **+32.6** |
| **compile rate** | 59.6% | **84.5%** | **+24.9** |

pass@1 by language pair × difficulty (%):

| source | target | difficulty | base | this model |
|---|---|---|---|---|
| cpp | java | Easy | 41.4 | 81.4 |
| cpp | java | Hard | 12.7 | 47.5 |
| cpp | java | Medium | 27.9 | 69.4 |
| cpp | python | Easy | 40.7 | 76.7 |
| cpp | python | Hard | 29.8 | 45.0 |
| cpp | python | Medium | 38.6 | 66.6 |
| java | cpp | Easy | 39.5 | 85.0 |
| java | cpp | Hard | 32.8 | 47.1 |
| java | cpp | Medium | 40.0 | 68.5 |
| java | python | Easy | 18.6 | 78.5 |
| java | python | Hard | 15.3 | 45.8 |
| java | python | Medium | 22.7 | 66.6 |
| python | cpp | Easy | 25.9 | 72.1 |
| python | cpp | Hard | 14.3 | 22.7 |
| python | cpp | Medium | 25.9 | 57.8 |
| python | java | Easy | 44.1 | 62.8 |
| python | java | Hard | 10.2 | 24.6 |
| python | java | Medium | 28.7 | 54.7 |

The base model also redefined the harness-provided `ListNode`/`TreeNode` helper types on ~6% of problems (a compile error); this fine-tune does so on **none**, having learned the dataset's convention. Full methodology is in the [llm-fine-tune](https://github.com/tkeskin/llm-fine-tune) repo (Stage 5).

## Intended use

Given source code in one of C++, Java, or Python, the model generates a translation into the target language, following the same logic and structure. The Qwen2.5-Coder base model includes code-specific pre-training across C, C++, Java, Python, and many other languages, giving it a stronger prior for code structure.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tkeskin/qwen2.5-coder-1.5b-code-translation"
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
