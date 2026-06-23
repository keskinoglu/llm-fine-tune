"""Update only the model card (README.md) on each published HF model repo.

The merged weights are already uploaded; this pushes just `model_card/<name>.md`
as the repo's README.md, so it's instant. Use after editing a card (e.g. adding
evaluation results). For a full model (re)upload, use `publish-model` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

_MODEL_CARD_DIR = Path(__file__).parent / "model_card"

# Card stem -> published repo id. The repo naming is irregular (llama carries an
# `-instruct-` infix the others don't), so the mapping is explicit rather than derived.
CARD_REPOS = {
    "qwen2.5-coder-1.5b": "tkeskin/qwen2.5-coder-1.5b-code-translation",
    "llama-3.2-1b": "tkeskin/llama-3.2-1b-instruct-code-translation",
    "qwen-3.5-0.8b": "tkeskin/qwen-3.5-0.8b-code-translation",
    "gemma-3-4b-it": "tkeskin/gemma-3-4b-it-code-translation",
    "mistral-7b-v0.3": "tkeskin/mistral-7b-v0.3-code-translation",
}


def main() -> None:
    args = _parse_args()
    api = HfApi()
    for card in args.cards or list(CARD_REPOS):
        repo_id = CARD_REPOS[card]
        card_path = _MODEL_CARD_DIR / f"{card}.md"
        if not card_path.exists():
            raise FileNotFoundError(f"No model card at {card_path}")
        print(f"Updating card: {card_path.name} -> {repo_id}")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=args.message,
        )
        print(f"  done: https://huggingface.co/{repo_id}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update model cards (README.md only) on the published HF model repos."
    )
    parser.add_argument(
        "--cards",
        nargs="*",
        choices=list(CARD_REPOS),
        help="Card stems to update (default: all).",
    )
    parser.add_argument(
        "--message",
        default="Update model card",
        help="Commit message for the card upload.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
