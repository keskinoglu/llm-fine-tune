# Stage 3: Fine-tuning

This directory contains everything needed to fine-tune a model using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) on an HPC cluster.

Each config fine-tunes one base model with **LoRA** on the `leetcode_instruct_train` dataset, holding out `leetcode_instruct_test` for evaluation. The configs under `configs/` ship as `<model>-lora.yaml` + `<model>-merge.yaml` pairs; the examples below use the Qwen-Coder run.

---

## Directory layout

```
finetune/
  configs/
    <model>-lora.yaml         — one LoRA config per model (AMD + NVIDIA, runs anywhere)
    <model>-merge.yaml        — matching export config for the merge step
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

### `<model>-lora.yaml` (the LoRA configs)

- `flash_attn: sdpa` — uses PyTorch's built-in SDPA. No `flash-attn` library needed. Works on
  AMD ROCm and NVIDIA CUDA alike.
- `bf16: true` — supported on AMD MI210 and NVIDIA Ampere+ (A100, H100, A6000). For older
  NVIDIA GPUs (V100, sm_70): set `bf16: false` and `fp16: true`.
- **Gated** base models (e.g. [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct))
  require accepting their license once on HuggingFace before your token will work. Apache-2.0 models
  like Qwen-Coder need no such acceptance.

### `gpt-oss-20b-lora.yaml` (advanced / NVIDIA Hopper-only)

LLaMA-Factory hardcodes FlashAttention-3 for `gpt_oss` and ignores the `flash_attn` config key.
FA3 requires NVIDIA Hopper (H100/H200) — it will not run on AMD or pre-Hopper NVIDIA. This config
is kept for reference; use any `<model>-lora.yaml` for anything that needs to run on AMD.

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
    src/llm_fine_tune/finetune/configs/qwen2.5-coder-1.5b-merge.yaml \
    "$WORK_DIR/saves/qwen2.5-coder-1.5b-lora" \
    tkeskin/qwen2.5-coder-1.5b-code-translation
```

The `<model>-merge.yaml` is the portable export config. The cluster job script injects
`adapter_name_or_path` and `export_dir` at runtime. If you omit the repo id argument, the script
merges only and prints the `publish-model` command to run manually.

See [`hpc/goethe/README.md`](hpc/goethe/README.md) for the full workflow including HF token requirements.

---

## Adding a new config

Drop a YAML in `configs/` and pass its path to the cluster's submit script. Common variants:

| File | Notes |
|---|---|
| `<model>-qlora.yaml` | Add `quantization_bit: 4`. Caution: bitsandbytes on ROCm is brittle. |
| `<model>-full.yaml` | Remove `finetuning_type: lora`. Full SFT needs DeepSpeed ZeRO-3. |
| `<model>-lora.yaml` | Set the right chat `template:` for the model family (e.g. `qwen`, `qwen3`). |

---

## How `dataset_info.json` works

LLaMA-Factory resolves datasets via the `dataset_dir` argument. Each job script points this at
`src/llm_fine_tune/finetune/`, where `dataset_info.json` lives. The entries there pull
`leetcode_instruct_train` and `leetcode_instruct_test` directly from HuggingFace Hub at job start
— no manual data copying. By default configs train on `leetcode_instruct_train` and evaluate on
`leetcode_instruct_test`; to train on everything, set `dataset: leetcode_instruct_train,leetcode_instruct_test`.

---

## Do not run `uv sync --extra rocm/cuda` locally

The finetune extras install hardware-specific PyTorch wheels. On a machine without the
corresponding GPU drivers, the wheel installs but `import torch` fails. Restrict these
extras to the cluster.
