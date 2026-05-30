#!/usr/bin/env bash
# Shared helpers sourced by every cluster job script.
# Source this AFTER the cluster's env.sh has loaded modules and set REPO_DIR/WORK_DIR.

# ---------------------------------------------------------------------------
# Validate required environment variables (fail fast with a readable message)
# ---------------------------------------------------------------------------
: "${WORK_DIR:?
  WORK_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export WORK_DIR=/work/<your_group>/<your_username>
}"
: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"

validate_config() {
    local config="$1"
    if [[ -z "$config" ]]; then
        echo "ERROR: No config file specified."
        echo "Usage: <submit-script> src/llm_fine_tune/finetune/configs/<name>.yaml"
        exit 1
    fi
    if [[ ! -f "$REPO_DIR/$config" ]]; then
        echo "ERROR: Config not found: $REPO_DIR/$config"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# launch_training <repo-relative-config> <output_dir> [extra key=value ...]
#
# Activates the venv, then runs llamafactory-cli train via torchrun.
# FORCE_TORCHRUN=1 makes LLaMA-Factory use torchrun, which spawns one process
# per visible GPU automatically — no manual --nproc_per_node needed.
# ---------------------------------------------------------------------------
launch_training() {
    local config="$1"
    local output_dir="$2"
    shift 2

    source "$REPO_DIR/.venv/bin/activate"

    echo "==> Starting training"
    echo "    Config:      $REPO_DIR/$config"
    echo "    Dataset dir: $REPO_DIR/src/llm_fine_tune/finetune"
    echo "    Output dir:  $output_dir"
    echo ""

    FORCE_TORCHRUN=1 llamafactory-cli train "$REPO_DIR/$config" \
        dataset_dir="$REPO_DIR/src/llm_fine_tune/finetune" \
        output_dir="$output_dir" \
        "$@"
}
