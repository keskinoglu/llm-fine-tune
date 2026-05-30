# Stage 3: Fine-tuning

This directory contains everything needed to fine-tune a model using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) on an HPC cluster.

The default config fine-tunes **[meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)** with **LoRA** on the `leetcode_instruct` dataset.

---

## Directory layout

```
finetune/
  configs/
    llama-3.2-1b-lora.yaml    — primary config (AMD + NVIDIA, runs anywhere)
    gpt-oss-20b-lora.yaml     — advanced / NVIDIA Hopper-only (see warning in file)
  hpc/
    common.sh                 — shared launch_training() + env validation (sourced by all clusters)
    goethe/                   — AMD MI210, ROCm, SLURM  →  see hpc/goethe/README.md
    saarland/                 — NVIDIA, CUDA, HTCondor  →  see hpc/saarland/README.md
  dataset_info.json           — LLaMA-Factory dataset registration (points to HF Hub)
  README.md                   — this file
```

---

## What's portable vs. cluster-specific

| Portable (shared) | Cluster-specific (under `hpc/<cluster>/`) |
|---|---|
| `configs/*.yaml` — model, hyperparams, method | Scheduler (SLURM `#SBATCH` / HTCondor `.sub`) |
| `dataset_info.json` — dataset pointer | GPU backend (`module load rocm` vs `module load cuda`) |
| `hpc/common.sh` — the training invocation itself | uv extra (`--extra rocm` vs `--extra cuda`) |

---

## Configs

### `llama-3.2-1b-lora.yaml` (default)

- `flash_attn: sdpa` — uses PyTorch's built-in SDPA. No `flash-attn` library needed. Works on
  AMD ROCm and NVIDIA CUDA alike.
- `bf16: true` — supported on AMD MI210 and NVIDIA Ampere+ (A100, H100, A6000). For older
  NVIDIA GPUs (V100, sm_70): set `bf16: false` and `fp16: true`.
- Requires accepting the [Meta Llama license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  once on HuggingFace before your token will work.

### `gpt-oss-20b-lora.yaml` (advanced / NVIDIA Hopper-only)

LLaMA-Factory hardcodes FlashAttention-3 for `gpt_oss` and ignores the `flash_attn` config key.
FA3 requires NVIDIA Hopper (H100/H200) — it will not run on AMD or pre-Hopper NVIDIA. This config
is kept for reference; use `llama-3.2-1b-lora.yaml` for anything that needs to run on AMD.

---

## Pick your cluster

- **AMD / ROCm / SLURM (Goethe)** → [`hpc/goethe/README.md`](hpc/goethe/README.md)
- **NVIDIA / CUDA / HTCondor (Saarland)** → [`hpc/saarland/README.md`](hpc/saarland/README.md)

---

## After training: merge and publish

LoRA training saves only the low-rank adapter deltas, not a standalone model. To get a model you can
actually use (and push to HuggingFace), merge the adapter into the base weights:

```bash
sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-merge.sh \
    src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml \
    "$WORK_DIR/saves/<run-name>" \
    tkeskin/llama-3.2-1b-instruct-code-translation
```

`configs/llama-3.2-1b-merge.yaml` is the portable export config. The cluster job script injects
`adapter_name_or_path` and `export_dir` at runtime. If you omit the repo id argument, the script
merges only and prints the `publish-model` command to run manually.

See [`hpc/goethe/README.md`](hpc/goethe/README.md) for the full workflow including HF token requirements.

---

## Adding a new config

Drop a YAML in `configs/` and pass its path to the cluster's submit script. Common variants:

| File | Notes |
|---|---|
| `llama-3.2-1b-qlora.yaml` | Add `quantization_bit: 4`. Caution: bitsandbytes on ROCm is brittle. |
| `llama-3.2-1b-full.yaml` | Remove `finetuning_type: lora`. Full SFT needs DeepSpeed ZeRO-3. |
| `qwen3-0.6b-lora.yaml` | Smaller/faster; use `template: qwen3`. |

---

## How `dataset_info.json` works

LLaMA-Factory resolves datasets via the `dataset_dir` argument. Each job script points this at
`src/llm_fine_tune/finetune/`, where `dataset_info.json` lives. The entry there pulls
`leetcode_instruct` directly from HuggingFace Hub at job start — no manual data copying.

---

## Do not run `uv sync --extra rocm/cuda` locally

The finetune extras install hardware-specific PyTorch wheels. On a machine without the
corresponding GPU drivers, the wheel installs but `import torch` fails. Restrict these
extras to the cluster.
