#!/usr/bin/env bash
# SLURM wrapper: build the bigcode execution sandbox and two generation venvs.
#
# Three things get created, all under $WORK_DIR:
#   images/bigcode-multiple   — Apptainer --sandbox; runs generated code in containers (CPU, no model)
#   eval-envs/bigcode-gen     — declarative uv env for bigcode generation (GPU, ROCm torch)
#   eval-envs/lmeval          — declarative uv env for lm-eval general tasks (GPU, ROCm torch)
#
# The bigcode image pull is the long step (~3h on slow links); hence partition=general1 + 6h wall.
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
BIGCODE_REPO="$WORK_DIR/bigcode-evaluation-harness"
IMAGES_DIR="$WORK_DIR/images"
BIGCODE_IMAGE="$IMAGES_DIR/bigcode-multiple"

mkdir -p "$IMAGES_DIR"

# ---------------------------------------------------------------------------
# Step 1: clone the bigcode-evaluation-harness repo (generation runs main.py directly,
# which the installed package does not expose; the env below installs bigcode_eval for imports)
# ---------------------------------------------------------------------------
if [[ ! -d "$BIGCODE_REPO" ]]; then
    echo "==> Cloning bigcode-evaluation-harness ..."
    git clone --depth 1 https://github.com/bigcode-project/bigcode-evaluation-harness.git "$BIGCODE_REPO"
else
    echo "==> bigcode-evaluation-harness already present at $BIGCODE_REPO, skipping clone."
fi

# ---------------------------------------------------------------------------
# Step 2: sync the two isolated, declarative uv envs (torch routed to the ROCm index
# inside each pyproject — one resolve, so no post-install pip clobber of the ROCm wheel)
# ---------------------------------------------------------------------------
echo ""
echo "==> Syncing eval-envs/bigcode-gen ..."
uv sync --project "$EVAL_ENVS_DIR/bigcode-gen"

echo "==> Syncing eval-envs/lmeval ..."
uv sync --project "$EVAL_ENVS_DIR/lmeval"

# ---------------------------------------------------------------------------
# Step 3: build the bigcode execution sandbox (Apptainer --sandbox, CPU-only)
# ---------------------------------------------------------------------------
# The prebuilt image (evaluation-harness-multiple) contains g++, javac, and
# the bigcode harness Python code — but no model weights and no GPU drivers.
# Execution-only (Phase 3 of submit-benchmark.sh) runs here with --net none.
#
# --sandbox (a directory, not a .sif): proot on this cluster can't run mksquashfs
# (nested ptrace denied), so we build a rootfs directory instead of a .sif.
echo ""
if [[ -d "$BIGCODE_IMAGE" ]]; then
    echo "==> $BIGCODE_IMAGE already exists, skipping build."
    echo "    (rm -rf it first to force a rebuild; re-running this script just re-syncs the envs.)"
else
    echo "==> Building $BIGCODE_IMAGE (Apptainer --sandbox from Docker image) ..."
    apptainer build --sandbox \
        "$BIGCODE_IMAGE" \
        "docker://ghcr.io/bigcode-project/evaluation-harness-multiple:latest"
fi

echo ""
echo "==> Benchmark setup complete!"
echo "    bigcode sandbox:    $BIGCODE_IMAGE"
echo "    bigcode-gen env:    $EVAL_ENVS_DIR/bigcode-gen (uv run --project)"
echo "    lmeval env:         $EVAL_ENVS_DIR/lmeval (uv run --project)"
echo "    bigcode repo:       $BIGCODE_REPO"
echo ""
echo "Smoke-test the sandbox (run these on any node with Apptainer):"
echo "    apptainer exec --net --network none '$BIGCODE_IMAGE' python3 /app/main.py --help"
echo "    apptainer exec --net --network none '$BIGCODE_IMAGE' g++ --version"
echo "    apptainer exec --net --network none '$BIGCODE_IMAGE' javac -version"
