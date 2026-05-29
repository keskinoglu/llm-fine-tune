#!/usr/bin/env bash
# One-time setup script for the Goethe cluster (AMD MI210 + ROCm + SLURM).
# Run this once after cloning the repo to your /work directory.
# Safe to re-run — all steps check whether they are already done.

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required environment variable
# ---------------------------------------------------------------------------
: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=/work/<your_group>/<your_username>/llm-fine-tune
}"

# ---------------------------------------------------------------------------
# Step 1: Install uv if not already present
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer puts uv in ~/.local/bin — add to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "==> uv already installed ($(uv --version))"
fi

# ---------------------------------------------------------------------------
# Step 2: Install all dependencies including LLaMA-Factory and PyTorch (ROCm)
#
# uv reads pyproject.toml and uv.lock and installs everything declared in
# the `finetune` dependency group. The ROCm torch wheel is pulled automatically
# via the [tool.uv.sources] + [[tool.uv.index]] config in pyproject.toml —
# no manual torch install needed.
#
# The .venv/ is created under $REPO_DIR, which must be on /work (not /home)
# because the venv with torch + transformers exceeds the 30 GB /home quota.
# ---------------------------------------------------------------------------
export UV_CACHE_DIR="$(dirname "$REPO_DIR")/.cache/uv"
export UV_LINK_MODE=hardlink
mkdir -p "$UV_CACHE_DIR"
echo "==> uv cache → $UV_CACHE_DIR"

echo "==> Installing dependencies (this downloads torch ROCm + transformers — ~10 min)..."
cd "$REPO_DIR"
uv sync --group finetune --no-progress

# ---------------------------------------------------------------------------
# Step 3: Authenticate with HuggingFace
#
# openai/gpt-oss-20b is a gated model. You must accept the license on
# https://huggingface.co/openai/gpt-oss-20b before the token will work.
# ---------------------------------------------------------------------------
echo ""
echo "==> Setup complete!"
echo ""
echo "Next: log in to HuggingFace so the training job can download gpt-oss-20b."
echo "Run these two commands:"
echo ""
echo "    source \$REPO_DIR/.venv/bin/activate"
echo "    hf auth login"
echo ""
echo "Paste your HF token when prompted. Verify with: hf auth whoami"
