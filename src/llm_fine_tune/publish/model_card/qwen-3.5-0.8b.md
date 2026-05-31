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
