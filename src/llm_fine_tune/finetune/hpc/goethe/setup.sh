#!/usr/bin/env bash
# One-time setup for the Goethe cluster (AMD MI210, ROCm, SLURM).
# Run via submit-setup.sh — not directly from the login node.
# Safe to re-run.

set -euo pipefail

: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"

# ---------------------------------------------------------------------------
# Load cluster environment
# ---------------------------------------------------------------------------
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
echo "==> uv cache → $UV_CACHE_DIR"

# ---------------------------------------------------------------------------
# Install uv if not present
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "==> uv already installed ($(uv --version))"
fi

# ---------------------------------------------------------------------------
# Install all dependencies (ROCm torch + LLaMA-Factory).
# The .venv is created under $REPO_DIR which must be on /work — the wheel
# + transformers together exceed the 30 GB /home quota.
# ---------------------------------------------------------------------------
echo "==> Installing dependencies (ROCm torch + LLaMA-Factory — ~10 min)..."
cd "$REPO_DIR"
uv sync --extra rocm --verbose

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "==> Setup complete!"
echo ""
echo "Next: log in to HuggingFace so the training job can download the model."
echo "Run these after this job finishes:"
echo ""
echo "    source \$REPO_DIR/.venv/bin/activate"
echo "    hf auth login"
echo "    hf auth whoami"
echo ""
echo "Also accept the model license in your browser before the token will work."
