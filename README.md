# llm-fine-tune

Fine-tune an open LLM to translate code between **C++**, **Java**, and **Python**.

The project is a pipeline of five stages. Each stage's detail lives with the thing it describes —
this README is the map and links out to the ground truth.

1. **[Build the dataset](#stage-1-build-the-dataset)** — Parse [`walkccc/LeetCode`](https://github.com/walkccc/LeetCode) into structured translation pairs and publish the [`tkeskin/leetcode-solutions`](https://huggingface.co/datasets/tkeskin/leetcode-solutions) HuggingFace dataset.
2. **[Pick a base model](#stage-2-pick-a-base-model)** — Compare tokenizer fertility across candidate models to choose the one that encodes code most efficiently.
3. **[Fine-tune](#stage-3-fine-tune)** — LoRA fine-tune the chosen base on the `instruct` config using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), on the Goethe-NHR cluster (AMD MI210).
4. **[Publish the model](#stage-4-publish-the-fine-tuned-model)** — Merge the LoRA adapter into the base weights and push the standalone model to HuggingFace (`tkeskin/<model>-code-translation`).
5. **[Evaluate](#stage-5-evaluate)** — Pull the base and the fine-tune from HuggingFace and run both, executing each translation against real test cases in a sandbox. The result is the **delta** between the two.

## Dataset

The dataset has two configurations — `base` (one row per LeetCode problem with per-language solution
columns) and `instruct` (directed translation triples, split 70/30 at problem granularity to prevent
leakage):

```python
from datasets import load_dataset

ds = load_dataset("tkeskin/leetcode-solutions", "base")
ds = load_dataset("tkeskin/leetcode-solutions", "instruct")
```

Full schema, column types, and per-language coverage live on the
[dataset card](https://huggingface.co/datasets/tkeskin/leetcode-solutions) (source in
[`dataset_card/README.md`](dataset_card/README.md)).

## Setup

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run `make help` for the full list of commands.

## Stage 1: Build the dataset

```bash
make dataset    # build base + instruct Parquet (clones walkccc/LeetCode on first run)
make publish    # build, then upload Parquet + dataset card to HuggingFace
```

Publishing needs a HuggingFace token (`export HF_TOKEN=hf_...` or `uv run huggingface-cli login`) and
uses the [Xet storage backend](https://huggingface.co/docs/hub/xet/using-xet-storage) automatically.
See `make help` for per-dataset and custom-message variants.

## Stage 2: Pick a base model

Before fine-tuning, compare how efficiently candidate models tokenize the dataset — fewer tokens per
unit of code means lower training and inference cost on the same budget. Only the tokenizer is loaded
from each model; weights are never downloaded.

```bash
make fertility   # evaluate the tokenizers listed in tokenizer-sources.txt
```

Configure which models to compare by editing [`tokenizer-sources.txt`](tokenizer-sources.txt)
(`<display-name>=<huggingface-model-id>` per line). Results print to stdout and save to
`output/tokenizer-fertility-report.parquet`.

Three metrics are reported per tokenizer — `tokens_per_word` (lower is better), `chars_per_token` and
`bytes_per_token` (higher is better). "Word" is defined by the regex `\w+`, so `snake_case` and
`camelCase` identifiers each count as one word, language-agnostically:

```
for(int i = 0; i < n; i++)   →  for  int  i  0  i  n  i      (7 words)
def calculate_total(items):  →  def  calculate_total  items  (3 words)
```

## Stage 3: Fine-tune

LoRA fine-tune the base model chosen in Stage 2 on the `instruct` dataset using
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Training configs are hardware-agnostic and
ship under [`finetune/configs/`](src/llm_fine_tune/finetune/configs/) (one `<model>-lora.yaml` +
`<model>-merge.yaml` pair per model); cluster-specific launch scripts live under
[`finetune/hpc/`](src/llm_fine_tune/finetune/hpc/).

See [`finetune/README.md`](src/llm_fine_tune/finetune/README.md) for the full workflow (config
structure, multi-GPU, ROCm vs CUDA). To push local config edits to the cluster without committing:

```bash
make finetune-sync   # requires CLUSTER_HOST and CLUSTER_REPO_DIR in .env
```

## Stage 4: Publish the fine-tuned model

LoRA training saves only the adapter deltas. This stage merges them into the base weights and pushes
the standalone model to HuggingFace (`tkeskin/<model>-code-translation`), where Stage 5 pulls it back.
The merge runs on the cluster (where the adapter, base cache, and venv live).

See [`finetune/hpc/goethe/README.md`](src/llm_fine_tune/finetune/hpc/goethe/README.md) for the merge +
publish workflow (SLURM job, HF token, version tagging), and
[`publish/model_card/`](src/llm_fine_tune/publish/model_card/README.md) for the model cards injected at
upload time.

## Stage 5: Evaluate

Measure whether the fine-tune produces translations that actually **compile and run** — not just
plausible-looking code. The workflow evaluates **two models pulled from HuggingFace** — the upstream
**base** and the Stage-4 **fine-tune** — and the conclusion is the **delta** between their `pass@1` /
`compiled` rates. Each runs on the cluster in three phases: generation (GPU), sandboxed execution of
untrusted model output (Apptainer, `--net --network none`), and reporting.

```bash
cd "$REPO_DIR"
make upload DATASET=evaluation   # publish the evaluation config first (run locally)

sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation-setup.sh   # one-time sandbox build
sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation.sh Qwen/Qwen2.5-Coder-1.5B-Instruct
sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation.sh tkeskin/qwen2.5-coder-1.5b-code-translation
```

A **standard-benchmark track** runs alongside the custom eval, comparing base vs fine-tune on
recognized benchmarks: held-out perplexity, HumanEval + MultiPL-E pass@1 via an in-house sandboxed
runner, and lm-eval general tasks (mmlu, gsm8k, hellaswag, arc_challenge, winogrande) to catch
catastrophic forgetting.

See [`evaluation/README.md`](src/llm_fine_tune/evaluation/README.md) for the full architecture,
metrics, the standard-benchmark track, and how to read the comparison.

---

> AI-DISCLAIMER: This codebase was AI-generated with minimal human review. Deploy with caution.
