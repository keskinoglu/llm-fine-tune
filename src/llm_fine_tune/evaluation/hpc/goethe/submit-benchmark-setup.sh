#!/usr/bin/env bash
# SLURM wrapper: sync the lm-eval generation venv.
#
# The standard-benchmark track needs one isolated Python env:
#   eval-envs/lmeval  — declarative uv env for lm-eval general tasks (GPU, ROCm torch)
#
# The code-benchmark (MultiPL-E) runner reuses the translation-eval Apptainer image
# (images/evaluation) already built by submit-evaluation-setup.sh — no separate image needed.
#
# The lmeval uv sync is heavy enough that the login node's watchdog can kill it, so we
# submit to partition=general1.
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-benchmark-setup.sh
#
# Required env vars (set in ~/.bashrc):
#   WORK_DIR  — e.g. /work/<group>/<user>
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune
#
# Monitor:
#   scontrol show job <JOBID> | grep StdOut
#   tail -f benchmark_setup_<JOBID>.out

#SBATCH --job-name=benchmark-setup
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --output=benchmark_setup_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

: "${WORK_DIR:?
  WORK_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export WORK_DIR=/work/<your_group>/<your_username>
}"
: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"

source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

EVAL_ENVS_DIR="$REPO_DIR/eval-envs"

# ---------------------------------------------------------------------------
# Sync the lm-eval env (GPU, ROCm torch)
# ---------------------------------------------------------------------------
echo "==> Syncing eval-envs/lmeval ..."
uv sync --project "$EVAL_ENVS_DIR/lmeval"

echo ""
echo "==> Benchmark setup complete!"
echo "    lmeval env: $EVAL_ENVS_DIR/lmeval (uv run --project)"
echo ""
echo "The code-benchmark (MultiPL-E) runner reuses the translation-eval execution sandbox."
echo "Make sure submit-evaluation-setup.sh has been run first so images/evaluation exists."
echo ""
echo "Smoke-test the execution sandbox:"
echo "    apptainer exec --net --network none '${WORK_DIR}/images/evaluation' g++ --version"
echo "    apptainer exec --net --network none '${WORK_DIR}/images/evaluation' javac -version"
echo "    apptainer exec --net --network none '${WORK_DIR}/images/evaluation' python3 --version"
