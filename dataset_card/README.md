---
license:
- mit
- apache-2.0
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
    path: leetcode-instruct-train.parquet
  - split: test
    path: leetcode-instruct-test.parquet
---

# LeetCode Solutions

Solutions to LeetCode problems in C++, Java, Python, SQL, and TypeScript, enriched
with problem metadata and test cases from a second source. Includes an
instruction-tuning variant for fine-tuning language models.

## Sources

Derived from two open-access sources:

- [walkccc/LeetCode](https://github.com/walkccc/LeetCode) by [@walkccc](https://github.com/walkccc), licensed under **MIT**.
- [newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset), licensed under **Apache 2.0**. Provides problem descriptions, difficulty labels, and input/output test cases.

Dataset built using [tkeskin/llm-fine-tune](https://github.com/tkeskin/llm-fine-tune).

## Configurations

### `base` (default)

One row per LeetCode problem. Language columns are `null` when no solution exists.
Metadata columns (`difficulty`, `input_output`, etc.) are `null` for problems not
present in the secondary source.

```python
from datasets import load_dataset
ds = load_dataset("tkeskin/leetcode-solutions", "base")
```

#### Columns

| Column              | Type   | Description                                                   |
|---------------------|--------|---------------------------------------------------------------|
| `problem_id`        | int64  | LeetCode problem number                                       |
| `title`             | string | Problem title                                                 |
| `cpp`               | string | C++ solution (~3,495 problems)                                |
| `java`              | string | Java solution (~3,371 problems)                               |
| `python`            | string | Python solution (~3,169 problems)                             |
| `sql`               | string | SQL solution (~307 problems)                                  |
| `typescript`        | string | TypeScript solution (~69 problems)                            |
| `difficulty`        | string | Problem difficulty: `Easy`, `Medium`, or `Hard`               |
| `input_output`      | list   | `[{"input": ..., "output": ...}]` test case pairs             |
| `problem_description` | string | Full problem statement                                      |
| `entry_point`       | string | Function/method name to implement                             |
| `prompt`            | string | Prompt template variant                                       |
| `query`             | string | Full problem prompt with context                              |
| `response`          | string | Reference explanation/response                                |
| `tags`              | list   | Topic tags (e.g. `["Array", "Hash Table"]`)                   |
| `estimated_date`    | date   | Problem publication date                                      |
| `task_id`           | string | URL slug identifier (e.g. `two-sum`)                          |

### `instruct`

Instruction-tuning triples derived from the `base` config. Each row is a directed
code-translation pair between C++, Java, and Python (e.g. Python→Java and Java→Python
are separate rows).

The dataset is split 70/30 at **problem granularity** — all translation pairs for a
given problem land on the same side, preventing train/test leakage. The split is
deterministic (seeded) for reproducibility.

```python
from datasets import load_dataset

# For training only (instructor's held-out eval method):
ds = load_dataset("tkeskin/leetcode-solutions", "instruct")
train = ds["train"]
test  = ds["test"]

# To train on everything (grade with your own metrics):
from datasets import concatenate_datasets
full = concatenate_datasets([ds["train"], ds["test"]])
```

| Column        | Description                                         |
|---------------|-----------------------------------------------------|
| `instruction` | Natural-language instruction (randomly varied)      |
| `input`       | Source code to translate from                       |
| `output`      | Target code to translate to                         |

## License

This dataset combines material from two sources under different licenses:

- Language solutions (`cpp`, `java`, `python`, `sql`, `typescript`) and `title` derive from
  [walkccc/LeetCode](https://github.com/walkccc/LeetCode), licensed under the **MIT License**.
- Problem metadata (`difficulty`, `input_output`, `problem_description`, `entry_point`,
  `prompt`, `query`, `response`, `tags`, `estimated_date`, `task_id`) derive from
  [newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset),
  licensed under the **Apache 2.0 License**.

Use of this dataset is subject to both licenses. LeetCode problem statements are the
intellectual property of LeetCode and are reproduced here for research purposes only.
