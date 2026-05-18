---
license: gpl-3.0
language:
- en
pretty_name: LeetCode Solutions
tags:
- code
- leetcode
- programming
configs:
- config_name: base
  default: true
  data_files:
  - split: train
    path: leetcode-solutions.parquet
- config_name: instruct
  data_files:
  - split: train
    path: leetcode-instruct.parquet
---

# LeetCode Solutions

Solutions to LeetCode problems in C++, Java, Python, SQL, and TypeScript, with an instruction-tuning variant for fine-tuning language models.

## Source

Derived from [walkccc/LeetCode](https://github.com/walkccc/LeetCode) by [@walkccc](https://github.com/walkccc), licensed under MIT.

Dataset built using [tkeskin/llm-fine-tune](https://github.com/tkeskin/llm-fine-tune).

## Configurations

### `base` (default)

One row per LeetCode problem. Language columns are `null` when no solution exists.

```python
from datasets import load_dataset
ds = load_dataset("tkeskin/leetcode-solutions", "base")
```

| Column       | Description                        |
|--------------|------------------------------------|
| `problem_id` | LeetCode problem number            |
| `title`      | Problem title                      |
| `cpp`        | C++ solution (~3,495 problems)     |
| `java`       | Java solution (~3,371 problems)    |
| `python`     | Python solution (~3,169 problems)  |
| `sql`        | SQL solution (~307 problems)       |
| `typescript` | TypeScript solution (~69 problems) |

### `instruct`

Instruction-tuning triples derived from the `base` config. Each row is a directed code-translation pair between C++, Java, and Python (e.g. Python→Java and Java→Python are separate rows).

```python
from datasets import load_dataset
ds = load_dataset("tkeskin/leetcode-solutions", "instruct")
```

| Column        | Description                                         |
|---------------|-----------------------------------------------------|
| `instruction` | Natural-language instruction (randomly varied)      |
| `input`       | Source code to translate from                       |
| `output`      | Target code to translate to                         |
