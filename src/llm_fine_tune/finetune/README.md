# Stage 3: Fine-tuning on the Goethe Cluster

This directory contains everything needed to fine-tune a model using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) on the **Goethe-NHR cluster** (AMD MI210 GPUs, ROCm, SLURM).

The default config fine-tunes **[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)** with **LoRA** on the `leetcode_instruct` dataset (the `instruct` split of [`tkeskin/leetcode-solutions`](https://huggingface.co/datasets/tkeskin/leetcode-solutions), pulled directly from HuggingFace Hub at training time).

---

## Directory layout

```
finetune/
  configs/
    gpt-oss-20b-lora.yaml   — LLaMA-Factory training config (add more here for new models/methods)
  scripts/
    cluster-setup.sh              — one-time install: uv, ROCm torch, LLaMA-Factory
    submit-setup.sh               — SLURM wrapper: runs cluster-setup.sh on a compute node
    submit-finetune-test.sh       — SLURM test job: 10 steps, 1 GPU, gpu_test partition
    submit-finetune.sh            — SLURM real job: all 8 GPUs, gpu partition, 8 hours
  dataset_info.json         — LLaMA-Factory dataset registration (points to HF Hub)
  README.md                 — this file
```

---

## One-time setup

**1. Set your work directory** in `~/.bashrc` and reload it:

```bash
export WORK_DIR=/work/<your_group>/<your_username>
```

```bash
source ~/.bashrc && echo $WORK_DIR
```

> **To make these permanent** (so they survive logout and `sbatch` sees them), append all vars to `~/.bashrc` in one go — running `export` at the prompt only lasts for the current session:
> ```bash
> cat >> ~/.bashrc <<'EOF'
>
> export WORK_DIR=/work/<your_group>/<your_username>
> export REPO_DIR=$WORK_DIR/llm-fine-tune
> export UV_CACHE_DIR=$WORK_DIR/.cache/uv
> export UV_LINK_MODE=hardlink
> EOF
> bash -l -c 'echo "$WORK_DIR $REPO_DIR $UV_CACHE_DIR"'
> ```
>
> `UV_CACHE_DIR` redirects uv's cache from `~/.cache/uv` (on the 30 GB `/home` quota) to `/work`. `UV_LINK_MODE=hardlink` avoids redundant copies when the cache and venv are on the same filesystem.

**2. Clone the repo into `$WORK_DIR`** — not `$HOME`. `/home` on Goethe is capped at ~30 GB; the venv + model weights need ~50 GB:

```bash
git clone https://github.com/keskinoglu/llm-fine-tune.git "$WORK_DIR/llm-fine-tune"
```

**3. Set `REPO_DIR`** in `~/.bashrc` to wherever you cloned it, then reload:

```bash
export REPO_DIR=$WORK_DIR/llm-fine-tune
```

```bash
source ~/.bashrc && echo $REPO_DIR
```

**4. Run the setup script** — installs uv, ROCm PyTorch, and LLaMA-Factory (~15 min, several GB).

The setup runs `uv sync` which extracts the ROCm torch wheel as a batch job on the `test` partition:

```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/scripts/submit-setup.sh
```

Monitor it:

```bash
squeue -u $USER
scontrol show job <JOBID> | grep StdOut   # get the exact log path
tail -f setup_<JOBID>.out
```

Look for `==> Setup complete!` at the end of the log.

**5. Log in to HuggingFace** — the setup script installs everything but does not log you in. Run these yourself after it finishes. `hf auth login` is what prompts you to paste your token.

You must also accept the model license at [huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) in your browser before the token will work.

```bash
source "$REPO_DIR/.venv/bin/activate"
hf auth login
hf auth whoami     # verify it worked
```

---

## Running a fine-tune job

**First: run the test job** (`gpu_test` partition with 1 GPU — finishes in ~15-30 min). This verifies the full pipeline before you commit to 8 hours:

```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/scripts/submit-finetune-test.sh \
    src/llm_fine_tune/finetune/configs/gpt-oss-20b-lora.yaml
```

Monitor it:

```bash
squeue -u $USER            # PD = pending, R = running, CG = completing
tail -f finetune_test_*.out
```

Look for these success markers in the log (in order):
1. `Loading checkpoint shards` — model weights loading
2. `trainable params:` — LoRA adapter initialized
3. `{'loss': X.XX, 'learning_rate': ...}` — training step completed
4. `max_steps reached` or `Training completed`

**Then: submit the real job** (all 8 GPUs, `gpu` partition, up to 8 hours):

```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/scripts/submit-finetune.sh \
    src/llm_fine_tune/finetune/configs/gpt-oss-20b-lora.yaml
```

```bash
squeue -u $USER
tail -f finetune_*.out
```

Checkpoints are saved every 200 steps under `$WORK_DIR/saves/gpt-oss-20b-lora/`.

---

## Adding a new config

Drop a YAML file in `configs/` and pass its path to `submit-finetune.sh`. Common variants:

| File | Notes |
|---|---|
| `gpt-oss-20b-qlora.yaml` | Add `quantization_bit: 4`. **Caution:** bitsandbytes on ROCm is brittle — prefer GPTQ or AWQ quantized weights instead. |
| `gpt-oss-20b-full.yaml` | Remove `finetuning_type: lora`. Full SFT needs DeepSpeed ZeRO-3: add `deepspeed: $REPO_DIR/LLaMA-Factory/examples/deepspeed/ds_z3_config.json`. |
| `gpt-oss-120b-lora.yaml` | Increase `gradient_accumulation_steps`, lower `per_device_train_batch_size`. Needs ZeRO-3 for model sharding across all 8 GPUs. |

---

## How `dataset_info.json` works

LLaMA-Factory resolves datasets via its `--dataset_dir` flag. The `submit-finetune.sh` script points this at `src/llm_fine_tune/finetune/`, where `dataset_info.json` lives. The entry there pulls `leetcode_instruct` directly from HuggingFace Hub at job start — no manual parquet copying.

---

## Do not run `uv sync --group finetune` locally

The `finetune` dependency group installs a ROCm-specific PyTorch wheel. On a machine without AMD GPU drivers, the wheel installs but `import torch` will fail. Restrict this group to the cluster.
