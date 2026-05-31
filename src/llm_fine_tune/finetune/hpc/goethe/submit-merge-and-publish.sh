#!/usr/bin/env bash
# Merge a LoRA adapter into the base model AND publish to HuggingFace.
# All arguments are required. To do steps separately:
#   merge only:   submit-merge.sh
#   publish only: publish-model --help
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-merge-and-publish.sh \
#       src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml \
#       "$WORK_DIR/saves/test-llama-3.2-1b-lora" \
#       tkeskin/llama-3.2-1b-instruct-code-translation \
#       v0.1 \
#       "10-step smoke test"
#
# Required env vars (set in ~/.bashrc):
#   WORK_DIR  — e.g. /work/<group>/<user>
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune

#SBATCH --job-name=llm-merge-publish
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:45:00
#SBATCH --output=merge_publish_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

CONFIG="${1:?
  Usage: sbatch submit-merge-and-publish.sh <config> <adapter_dir> <repo_id> <tag> <message>
  Missing: config  (e.g. src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml)
}"
ADAPTER_DIR="${2:?
  Usage: sbatch submit-merge-and-publish.sh <config> <adapter_dir> <repo_id> <tag> <message>
  Missing: adapter_dir  (e.g. \$WORK_DIR/saves/test-llama-3.2-1b-lora)
}"
REPO_ID="${3:?
  Usage: sbatch submit-merge-and-publish.sh <config> <adapter_dir> <repo_id> <tag> <message>
  Missing: repo_id  (e.g. tkeskin/llama-3.2-1b-instruct-code-translation)
}"
TAG="${4:?
  Usage: sbatch submit-merge-and-publish.sh <config> <adapter_dir> <repo_id> <tag> <message>
  Missing: tag  (e.g. v0.1)
}"
MESSAGE="${5:?
  Usage: sbatch submit-merge-and-publish.sh <config> <adapter_dir> <repo_id> <tag> <message>
  Missing: message  (e.g. \"10-step smoke test\")
}"

module load rocm/6.2.4
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

validate_config "$CONFIG"

EXPORT_DIR="$WORK_DIR/exports/$(basename "$ADAPTER_DIR")"

merge_lora "$CONFIG" "$ADAPTER_DIR" "$EXPORT_DIR"

echo ""
echo "==> Publishing merged model to $REPO_ID"
publish-model \
    --model-dir "$EXPORT_DIR" \
    --repo-id "$REPO_ID" \
    --tag "$TAG" \
    --message "$MESSAGE"
