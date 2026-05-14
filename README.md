# llm-fine-tune

Tools to build and publish the [`tkeskin/leetcode-solutions`](https://huggingface.co/datasets/tkeskin/leetcode-solutions) HuggingFace dataset from the [`walkccc/LeetCode`](https://github.com/walkccc/LeetCode) repository.

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

The dataset card (displayed on HuggingFace) lives in [`dataset_card/README.md`](dataset_card/README.md).

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Building the dataset

```bash
make dataset
```

On first run this clones `walkccc/LeetCode` into `data/leetcode-source/` (~12 MB), then writes `output/leetcode-solutions.parquet`. Subsequent runs reuse the existing clone. Pass `--pull` to fetch the latest changes first:

```bash
uv run python -m llm_fine_tune.build_dataset --pull
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

Then upload the Parquet and dataset card in a single commit:

```bash
make upload
```

Or build and upload in one shot:

```bash
make publish
```

To customise the commit message:

```bash
uv run python -m llm_fine_tune.upload_dataset --message "Add new solutions"
```

Uploads use HuggingFace's [Xet storage backend](https://huggingface.co/docs/hub/xet/using-xet-storage) automatically via `huggingface-hub>=0.32.0`.

## Available commands

```
make lint        Check code with ruff linting and format checks
make lf          Fix code with ruff linting and formatting
make commit      Run lint checks, then create a commitizen commit
make cz          Alias for 'make commit'
make bump        Bump the project version using commitizen
make dataset     Clone walkccc/LeetCode (if needed) and build the Parquet dataset
make clean-data  Remove the cloned source repo and generated output
make upload      Upload existing Parquet + dataset card to HuggingFace
make publish     Build the dataset, then upload it (combines dataset + upload)
```
