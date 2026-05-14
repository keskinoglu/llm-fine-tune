# llm-fine-tune

Tools to build the [`tkeskin/leetcode-solutions`](https://huggingface.co/datasets/tkeskin/leetcode-solutions) HuggingFace dataset from the [`walkccc/LeetCode`](https://github.com/walkccc/LeetCode) repository.

## Dataset schema

Each row is one LeetCode problem. Language columns contain the solution source code and are `null` when no solution exists for that language.

| Column       | Type   | Description                        |
|--------------|--------|------------------------------------|
| `problem_id` | int64  | LeetCode problem number            |
| `title`      | string | Problem title                      |
| `cpp`        | string | C++ solution (~3,495 problems)     |
| `java`       | string | Java solution (~3,371 problems)    |
| `python`     | string | Python solution (~3,169 problems)  |
| `sql`        | string | SQL solution (~307 problems)       |
| `typescript` | string | TypeScript solution (~69 problems) |

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
make dataset
```

On first run this clones `walkccc/LeetCode` into `data/leetcode-source/` (~12 MB), then writes `output/leetcode-solutions.parquet`. Subsequent runs reuse the existing clone.

To start fresh:

```bash
make clean-data
make dataset
```

## Uploading to HuggingFace

Copy the generated Parquet file into your cloned HF dataset repo and push:

```bash
cp output/leetcode-solutions.parquet ~/hf-datasets/leetcode-solutions/
cd ~/hf-datasets/leetcode-solutions
git add leetcode-solutions.parquet
git commit -m "Update dataset"
git push
```

## Available commands

```
make lint        Check code with ruff linting and format checks
make lf          Fix code with ruff linting and formatting
make commit      Run lint checks, then create a commitizen commit
make cz          Alias for 'make commit'
make bump        Bump the project version using commitizen
make dataset     Clone walkccc/LeetCode (if needed) and build the Parquet dataset
make clean-data  Remove the cloned source repo and generated output
```
