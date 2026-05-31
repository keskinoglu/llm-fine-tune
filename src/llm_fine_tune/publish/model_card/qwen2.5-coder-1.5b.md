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
