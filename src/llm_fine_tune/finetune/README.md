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
    cluster-setup.sh        — one-time install: uv, ROCm torch, LLaMA-Factory
    submit-finetune.sh      — SLURM batch script; takes a config path as argument
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

**4. Run the setup script** — installs uv, ROCm PyTorch, and LLaMA-Factory (~10 min, several GB):

```bash
bash "$REPO_DIR/src/llm_fine_tune/finetune/scripts/cluster-setup.sh"
```

**5. Log in to HuggingFace** (required to download the gated `gpt-oss-20b` weights):

```bash
source "$REPO_DIR/.venv/bin/activate"
huggingface-cli login      # paste your HF token when prompted
huggingface-cli whoami     # verify it worked
```

---

## Running a fine-tune job

```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/scripts/submit-finetune.sh \
    src/llm_fine_tune/finetune/configs/gpt-oss-20b-lora.yaml
```

Monitor the job:

```bash
squeue -u $USER            # see queue status (PD = pending, R = running)
sqtimes                    # estimated start time
tail -f finetune_*.out     # stream the training log
```

Checkpoints are saved under `$WORK_DIR/saves/gpt-oss-20b-lora/`.

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
