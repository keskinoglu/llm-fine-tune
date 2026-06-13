# Goethe Cluster — AMD MI210, ROCm, SLURM

This is the cluster-specific setup guide for the **Goethe-NHR cluster** (8× AMD MI210 GPUs per node,
ROCm, SLURM job scheduler).

For the portable training configs and an overview of the full directory layout, see
[`../../README.md`](../../README.md).

---

## One-time setup

**1. Set your work directory** permanently in `~/.bashrc` so it survives logout and is visible in
SLURM batch jobs:

```bash
cat >> ~/.bashrc <<'EOF'

export WORK_DIR=/work/<your_group>/<your_username>
export REPO_DIR=$WORK_DIR/llm-fine-tune
export UV_CACHE_DIR=$WORK_DIR/.cache/uv
export UV_LINK_MODE=hardlink
export HF_HOME=$WORK_DIR/.cache/huggingface
EOF
bash -l -c 'echo "$WORK_DIR $REPO_DIR $UV_CACHE_DIR $HF_HOME"'
```

`UV_CACHE_DIR` and `HF_HOME` redirect caches from `~/.cache/` (30 GB `/home` quota) to `/work`.
Without `HF_HOME`, large models (4B+) will exceed the home quota during download.
`UV_LINK_MODE=hardlink` avoids redundant copies when cache and venv share the filesystem.

If you set `HF_HOME` after already downloading models, move the existing cache to avoid re-downloading:

```bash
mkdir -p $WORK_DIR/.cache/huggingface
mv ~/.cache/huggingface/hub $WORK_DIR/.cache/huggingface/hub
```

**2. Clone the repo to `$WORK_DIR`** — not `$HOME`. `/home` is capped at ~30 GB; the venv + model
weights need ~50 GB:

```bash
git clone https://github.com/keskinoglu/llm-fine-tune.git "$WORK_DIR/llm-fine-tune"
```

**3. Run the setup job** — installs uv, ROCm PyTorch, and LLaMA-Factory (~15 min). This runs
`uv sync` on the `test` compute partition because the login-node CPU-time limit kills heavy processes
silently:

```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-setup.sh
```

Monitor it:

```bash
squeue -u $USER
scontrol show job <JOBID> | grep StdOut
tail -f setup_<JOBID>.out
```

Look for `==> Setup complete!` at the end.

**4. Log in to HuggingFace** — the setup job installs everything but does not log you in. Run these
yourself after it finishes. You must also accept the model license on
[huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
in your browser before the token will work:

```bash
source "$REPO_DIR/.venv/bin/activate"
hf auth login
hf auth whoami
```

---

## Running a fine-tune job

**First: run the test job** (1 GPU, `gpu_test` partition, 10 steps, ~15-30 min). This verifies the
full pipeline before committing to 8 hours:

```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-test.sh \
    src/llm_fine_tune/finetune/configs/llama-3.2-1b-lora.yaml
```

Monitor:

```bash
squeue -u $USER
tail -f finetune_test_*.out
```

Success markers (in order):
1. `Loading checkpoint shards` — model weights loading
2. `Using torch SDPA` — attention confirmed (not FlashAttention)
3. `trainable params:` — LoRA adapter initialized
4. `{'loss': X.XX, 'learning_rate': ...}` — training step completed

**Then: submit the full job** (8 GPUs, `gpu` partition, up to 8 hours):

```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/hpc/goethe/submit.sh \
    src/llm_fine_tune/finetune/configs/llama-3.2-1b-lora.yaml
```

Checkpoints are saved under `$WORK_DIR/saves/llama-3.2-1b-lora/`.

---

## Merging and publishing the fine-tuned model

After training, the saved directory contains only the LoRA adapter deltas. The merge step fuses
those deltas into the base model weights to produce a self-contained model you can push to HuggingFace.

**Prerequisite: HF write token.** The token cached via `hf auth login` may only have read access
(sufficient for downloading the gated Llama model). Pushing a model requires write access to
`tkeskin/` repos. If the push returns 401/403, re-run `hf auth login` with a write-scoped token.

Three separate commands — use the one that fits:

**1. Merge only** — fuses the adapter into the base model, writes to `$WORK_DIR/exports/<run>/`:
```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-merge.sh \
    src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml \
    "$WORK_DIR/saves/test-llama-3.2-1b-lora"
```

**2. Publish only** — uploads an already-merged directory to HuggingFace (run from login node after activating the venv):
```bash
source "$REPO_DIR/.venv/bin/activate"
publish-model \
    --model-dir "$WORK_DIR/exports/test-llama-3.2-1b-lora" \
    --repo-id tkeskin/llama-3.2-1b-instruct-code-translation \
    --card llama-3.2-1b \
    --tag v0.1 \
    --message "10-step smoke test"
```

**3. Merge and publish** — first five arguments required, sixth (card) optional:
```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-merge-and-publish.sh \
    src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml \
    "$WORK_DIR/saves/test-llama-3.2-1b-lora" \
    tkeskin/llama-3.2-1b-instruct-code-translation \
    v0.1 \
    "10-step smoke test" \
    llama-3.2-1b
```

Monitor any of these jobs with:
```bash
tail -f merge_*.out        # submit-merge.sh
tail -f merge_publish_*.out  # submit-merge-and-publish.sh
```

On success the log ends with `Done! https://huggingface.co/tkeskin/llama-3.2-1b-instruct-code-translation`.

---

## Cluster quick-reference

**Watch the job queue:**
```bash
watch -n 1 'squeue -u $USER'
```
Status codes: `PD` pending, `R` running, `CG` completing, `F` failed, `CA` cancelled, `TO` timed out

**See when a pending job will start, or where it sits in the queue:**
```bash
squeue --start -j <JOBID>            # SLURM's estimated start time for one pending job
squeue -p <partition> -t PD --sort=p # all pending jobs on a partition, highest priority first
```

**Get the log path for a running job:**
```bash
scontrol show job <JOBID> | grep StdOut
```

**Tail a job log:**
```bash
tail -f <path/to/logfile>.out
```

**Rebuild the venv after a lock-file change** (e.g. after bumping a package version locally and pushing):
```bash
cd "$REPO_DIR"
sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-rebuild-env.sh
```
This pulls the latest `pyproject.toml` / `uv.lock` from the remote and runs `uv sync --extra rocm`.
The log ends with the installed torch version so you can confirm the upgrade.

**Wipe and rebuild the venv from scratch** (if the install is corrupt):
```bash
sbatch --partition=test --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32g --time=00:30:00 \
  --output=setup_%j.out \
  --wrap="source ~/.bashrc && cd \$REPO_DIR && git pull && uv venv --clear && uv sync --extra rocm --verbose"
```

**Reinstall a single package** (e.g. after fixing a version pin without a full rebuild):
```bash
sbatch --partition=test --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16g --time=00:15:00 \
  --output=setup_%j.out \
  --wrap="source ~/.bashrc && cd \$REPO_DIR && uv sync --extra rocm --verbose --reinstall-package <package-name>"
```

**Cancel a job:**
```bash
scancel <JOBID>
```
