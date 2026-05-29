#!/usr/bin/env bash
# SLURM batch script for fine-tuning on the Goethe cluster (AMD MI210, gpu partition).
#
# Usage:
#   sbatch submit-finetune.sh src/llm_fine_tune/finetune/configs/gpt-oss-20b-lora.yaml
#
# Required environment variables (set in ~/.bashrc on the cluster):
#   WORK_DIR   — your fast work directory, e.g. /work/<group>/<username>
#   REPO_DIR   — path to this cloned repo, e.g. $WORK_DIR/llm-fine-tune
#
# Output logs land in finetune_<jobid>.out in the submission directory.

#SBATCH --job-name=llm-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150g
#SBATCH --time=08:00:00
#SBATCH --output=finetune_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
: "${WORK_DIR:?
  WORK_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export WORK_DIR=/work/<your_group>/<your_username>
}"
: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"
CONFIG="${1:?
  No config file specified.
  Usage: sbatch submit-finetune.sh <path/to/config.yaml>
  Example: sbatch submit-finetune.sh src/llm_fine_tune/finetune/configs/gpt-oss-20b-lora.yaml
}"

if [[ ! -f "$REPO_DIR/$CONFIG" ]]; then
    echo "ERROR: Config not found: $REPO_DIR/$CONFIG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module load rocm/6.2.4
source "$REPO_DIR/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Launch training
#
# FORCE_TORCHRUN=1 tells LLaMA-Factory to launch the training process via
# `torchrun` instead of plain `python`. On the `gpu` partition we are
# allocated a full node with 8 AMD MI210 GPUs, and torchrun automatically
# spawns one process per visible GPU — no manual --nproc_per_node needed.
# Without this flag, training runs single-GPU regardless of allocation.
# ---------------------------------------------------------------------------
OUTPUT_DIR="$WORK_DIR/saves/$(basename "$CONFIG" .yaml)"
echo "==> Starting fine-tune job"
echo "    Config:     $REPO_DIR/$CONFIG"
echo "    Dataset dir: $REPO_DIR/src/llm_fine_tune/finetune"
echo "    Output dir: $OUTPUT_DIR"
echo ""

FORCE_TORCHRUN=1 llamafactory-cli train "$REPO_DIR/$CONFIG" \
    dataset_dir="$REPO_DIR/src/llm_fine_tune/finetune" \
    output_dir="$OUTPUT_DIR"
