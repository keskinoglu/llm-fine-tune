# llm-fine-tune

Tools to build and publish the [`tkeskin/leetcode-solutions`](https://huggingface.co/datasets/tkeskin/leetcode-solutions) HuggingFace dataset from the [`walkccc/LeetCode`](https://github.com/walkccc/LeetCode) repository.

The dataset has two configurations:

```python
from datasets import load_dataset

# One row per LeetCode problem with per-language solution columns
ds = load_dataset("tkeskin/leetcode-solutions", "base")

# Instruction-tuning triples for directed code translation
ds = load_dataset("tkeskin/leetcode-solutions", "instruct")
```

## Dataset schema

### `base`

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

### `instruct`

Derived from `base`. Each row is a directed code-translation pair between C++, Java, and Python (Python→Java and Java→Python are separate rows).

| Column        | Type   | Description                                    |
|---------------|--------|------------------------------------------------|
| `instruction` | string | Natural-language instruction (randomly varied) |
| `input`       | string | Source code to translate from                  |
| `output`      | string | Target code to translate to                    |

The dataset card (displayed on HuggingFace) lives in [`dataset_card/README.md`](dataset_card/README.md).

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Building the datasets

Build the `base` dataset (clones `walkccc/LeetCode` on first run):

```bash
make base
```

Pass `--pull` to fetch the latest changes before building:

```bash
uv run python -m llm_fine_tune.build_base_dataset --pull
```

Build the `instruct` dataset from the `base` parquet:

```bash
make instruct
```

Build both in one shot:

```bash
make dataset
```

To start fresh:

```bash
make clean-data
make dataset
```

## Uploading to HuggingFace

Authentication is required. Set your HF token once:

```bash
export HF_TOKEN="hf_..."
```

Or store it permanently via:

```bash
uv run huggingface-cli login
```

Then upload both Parquet files and the dataset card in a single atomic commit:

```bash
make upload
```

Or build and upload in one shot:

```bash
make publish
```

To customise the commit message:

```bash
uv run python -m llm_fine_tune.upload_dataset --message "Add instruct config"
```

Uploads use HuggingFace's [Xet storage backend](https://huggingface.co/docs/hub/xet/using-xet-storage) automatically via `huggingface-hub>=0.32.0`.

## Available commands

```
make lint        Check code with ruff linting and format checks
make lf          Fix code with ruff linting and formatting
make commit      Run lint checks, then create a commitizen commit
make cz          Alias for 'make commit'
make bump        Bump the project version using commitizen
make base        Clone walkccc/LeetCode (if needed) and build the base Parquet dataset
make instruct    Build the instruct Parquet dataset from the base dataset
make dataset     Build both the base and instruct datasets
make clean-data  Remove the cloned source repo and generated output
make upload      Upload existing Parquet files + dataset card to HuggingFace
make publish     Build both datasets, then upload them (combines dataset + upload)
```
