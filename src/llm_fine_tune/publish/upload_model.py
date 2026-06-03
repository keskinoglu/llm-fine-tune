"""Publish a fine-tuned model to HuggingFace Hub (Stage 4).

Takes a locally merged model directory (output of llamafactory-cli export),
creates or reuses the target HF repo, injects an optional model card, uploads
the folder, and applies an optional git tag.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from llm_fine_tune import loaders

DEFAULT_REPO_ID = "tkeskin/llama-3.2-1b-instruct-code-translation"
DEFAULT_COMMIT_MESSAGE = "Upload merged model"
_MODEL_CARD_DIR = Path(__file__).parent / "model_card"


def main() -> None:
    args = _parse_args()

    loaders.require_file(
        args.model_dir, "run the merge job first: sbatch .../submit-merge.sh"
    )

    if args.card:
        _inject_model_card(args.model_dir, args.card)

    api = HfApi()
    _create_repo(api, args.repo_id, private=args.private)
    _upload_model(api, args.model_dir, args.repo_id, args.message)

    if args.tag:
        _apply_tag(api, args.repo_id, args.tag)

    print(f"\nDone! https://huggingface.co/{args.repo_id}")


# ---- Argument parsing ----


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a merged fine-tuned model to HuggingFace."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Path to the merged model directory (output of llamafactory-cli export).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo id to push to (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional git tag to apply after upload, e.g. v1.",
    )
    parser.add_argument(
        "--card",
        default=None,
        metavar="NAME",
        help=(
            "Model card to inject as README.md (e.g. 'llama-3.2-1b' → model_card/llama-3.2-1b.md). "
            "If omitted, no card is written."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public).",
    )
    return parser.parse_args()


# ---- Model card ----


def _inject_model_card(model_dir: Path, card_name: str) -> None:
    card_path = _MODEL_CARD_DIR / f"{card_name}.md"
    if not card_path.exists():
        available = [path.stem for path in sorted(_MODEL_CARD_DIR.glob("*.md"))]
        raise FileNotFoundError(
            f"No model card '{card_name}.md' in {_MODEL_CARD_DIR}\n"
            f"Available: {available}"
        )
    shutil.copy(card_path, model_dir / "README.md")


# ---- HuggingFace Hub ----


def _create_repo(api: HfApi, repo_id: str, *, private: bool) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    print(f"Repo ready: https://huggingface.co/{repo_id}")


def _upload_model(api: HfApi, model_dir: Path, repo_id: str, message: str) -> None:
    print(f"Uploading {model_dir} → {repo_id} ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
    )


def _apply_tag(api: HfApi, repo_id: str, tag: str) -> None:
    api.create_tag(repo_id=repo_id, repo_type="model", tag=tag)
    print(f"Tagged: {tag}")


if __name__ == "__main__":
    main()
