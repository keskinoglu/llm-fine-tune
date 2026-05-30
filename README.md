# llm-fine-tune

Fine-tune an open LLM to translate code between **C++**, **Java**, and **Python**.

The project is organized as a pipeline of four stages:

1. **Build the dataset** — Parse [`walkccc/LeetCode`](https://github.com/walkccc/LeetCode) into structured translation pairs and publish them as the [`tkeskin/leetcode-solutions`](https://huggingface.co/datasets/tkeskin/leetcode-solutions) HuggingFace dataset.
2. **Pick a base model** — Compare tokenizer fertility across candidate HuggingFace models to choose the one that encodes code most efficiently.
3. **Fine-tune** — Fine-tune the chosen base model on the `instruct` configuration using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) on the Goethe-NHR cluster (AMD MI210 GPUs).
4. **Publish the model** — Merge the LoRA adapter into the base weights and push the standalone fine-tuned model to [`tkeskin/llama-3.2-1b-instruct-code-translation`](https://huggingface.co/tkeskin/llama-3.2-1b-instruct-code-translation) on HuggingFace.

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

## Stage 1: Build the dataset

Pass `--pull` to fetch the latest changes before building:

```bash
uv run python -m llm_fine_tune.dataset.build_base_dataset --pull
```

Build the `base` dataset (clones `walkccc/LeetCode` on first run):

```bash
make base
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

### Publishing to HuggingFace

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
uv run python -m llm_fine_tune.dataset.upload_dataset --message "Add instruct config"
```

Uploads use HuggingFace's [Xet storage backend](https://huggingface.co/docs/hub/xet/using-xet-storage) automatically via `huggingface-hub>=0.32.0`.

## Stage 2: Pick a base model

Before fine-tuning, compare how efficiently candidate HuggingFace models tokenize the dataset. A tokenizer that produces fewer tokens per unit of code means lower fine-tuning and inference cost on the same training budget.

Tokenizer fertility measures how efficiently a tokenizer encodes text. Lower `tokens_per_word` and higher `bytes_per_token` mean better compression — especially important for code-heavy datasets.

Run the evaluation against `output/leetcode-instruct.parquet` (build it first with `make instruct`):

```bash
make fertility
```

The script loads only the tokenizer from each model — weights are never downloaded. Results are printed to stdout and saved to `output/tokenizer-fertility-report.parquet`.

Three metrics are reported per tokenizer:

| Metric | What it measures | Better when |
|---|---|---|
| `tokens_per_word` | Tokens per word in the code | Lower |
| `chars_per_token` | Characters compressed per token | Higher |
| `bytes_per_token` | UTF-8 bytes per token (compression ratio) | Higher |

Typical ranges for code: `tokens_per_word` ~1.5–5, `bytes_per_token` ~2.5–4.

**How "word" is defined:** words are extracted with the regex `\w+` (`[a-zA-Z0-9_]`), which splits on punctuation, operators, and whitespace. This is language-agnostic — it works identically for Python, Java, C++, SQL, etc. The key coding implication is that `_` is included, so `snake_case` identifiers count as **one** word, which matches how developers read them. `camelCase` also counts as one word (no sub-word splitting). Numbers count as words too.

```
for(int i = 0; i < n; i++)   →  for  int  i  0  i  n  i   (7 words)
def calculate_total(items):   →  def  calculate_total  items  (3 words)
```

### Configuring which tokenizers to evaluate

Edit `tokenizer-sources.txt` at the project root. Each line maps a display name to a HuggingFace model ID:

```
# Each line: <display-name>=<huggingface-model-id>
# Only the tokenizer is loaded — model weights are never downloaded.
# Lines starting with '#' and blank lines are ignored.
qwen2.5-coder-7b=Qwen/Qwen2.5-Coder-7B-Instruct
codestral-22b=mistralai/Codestral-22B-v0.1
```

To evaluate against a remote HuggingFace dataset instead of the local parquet:

```bash
uv run python -m llm_fine_tune.tokenizer.analyze_tokenizer_fertility \
  --hf-dataset tkeskin/leetcode-solutions \
  --hf-config instruct
```

For a quick smoke test on a subset:

```bash
uv run python -m llm_fine_tune.tokenizer.analyze_tokenizer_fertility --limit 100
```

Use a custom tokenizer sources file:

```bash
uv run python -m llm_fine_tune.tokenizer.analyze_tokenizer_fertility -s my-tokenizers.txt
```

## Stage 3: Fine-tune

Fine-tunes the base model chosen in Stage 2 on the `instruct` dataset using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) on an HPC cluster.

The default config fine-tunes **[meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)** with **LoRA** using `flash_attn: sdpa` — runs on AMD ROCm and NVIDIA CUDA without any FlashAttention library. Multi-GPU training via `torchrun` is automatic.

The training configs are hardware-agnostic. Cluster-specific scripts live under `hpc/`:
- **AMD / ROCm / SLURM (Goethe):** [`hpc/goethe/`](src/llm_fine_tune/finetune/hpc/goethe/)
- **NVIDIA / CUDA / HTCondor (Saarland):** [`hpc/saarland/`](src/llm_fine_tune/finetune/hpc/saarland/)

See [`src/llm_fine_tune/finetune/README.md`](src/llm_fine_tune/finetune/README.md) for the full overview.

To push config edits to the cluster without committing:

```bash
make finetune-sync   # requires CLUSTER_HOST and CLUSTER_REPO_DIR in .env
```

## Stage 4: Publish the fine-tuned model

LoRA training saves only the adapter deltas — not a standalone model. The publish stage merges those
deltas into the base weights and pushes the result to HuggingFace as
[`tkeskin/llama-3.2-1b-instruct-code-translation`](https://huggingface.co/tkeskin/llama-3.2-1b-instruct-code-translation).

The merge runs on the cluster (where the adapter, base model cache, and venv live). The publish step
uses the `publish-model` entry point, which handles repo creation, model card injection, upload, and
optional version tagging.

**Merge + publish in one SLURM job (Goethe):**
```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-merge.sh \
    src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml \
    "$WORK_DIR/saves/<run-name>" \
    tkeskin/llama-3.2-1b-instruct-code-translation
```

**Re-publish an already-merged model** (e.g. after a full training run, or to add a version tag):
```bash
source "$REPO_DIR/.venv/bin/activate"
publish-model \
    --model-dir "$WORK_DIR/exports/<run-name>" \
    --repo-id tkeskin/llama-3.2-1b-instruct-code-translation \
    --message "Fully trained v1 (3 epochs)" \
    --tag v1
```

See [`src/llm_fine_tune/finetune/hpc/goethe/README.md`](src/llm_fine_tune/finetune/hpc/goethe/README.md)
for the full merge + publish workflow including HF token requirements.

---

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
make fertility       Compute tokenizer fertility for sources in tokenizer-sources.txt
make finetune-sync   Rsync finetune/ configs and scripts to the cluster
publish-model        Merge + push: upload a merged model directory to HuggingFace
```
