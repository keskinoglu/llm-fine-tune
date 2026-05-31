from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "tkeskin/llama-3.2-1b-instruct-code-translation"
DEFAULT_COMMIT_MESSAGE = "Upload merged model"
_MODEL_CARD_DIR = Path(__file__).parent / "model_card"


def _inject_model_card(model_dir: Path, card_name: str) -> None:
    src = _MODEL_CARD_DIR / f"{card_name}.md"
    if not src.exists():
        available = [p.stem for p in sorted(_MODEL_CARD_DIR.glob("*.md"))]
        raise FileNotFoundError(
            f"No model card '{card_name}.md' in {_MODEL_CARD_DIR}\n"
            f"Available: {available}"
        )
    shutil.copy(src, model_dir / "README.md")


def _create_repo(api: HfApi, repo_id: str, private: bool) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    print(f"Repo ready: https://huggingface.co/{repo_id}")


def _upload(api: HfApi, model_dir: Path, repo_id: str, message: str) -> None:
    print(f"Uploading {model_dir} → {repo_id} ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
    )


def _tag(api: HfApi, repo_id: str, tag: str) -> None:
    api.create_tag(repo_id=repo_id, repo_type="model", tag=tag)
    print(f"Tagged: {tag}")


def main() -> None:
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
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}\n"
            "Run the merge job first: sbatch .../submit-merge.sh"
        )

    if args.card:
        _inject_model_card(args.model_dir, args.card)

    api = HfApi()
    _create_repo(api, args.repo_id, args.private)
    _upload(api, args.model_dir, args.repo_id, args.message)

    if args.tag:
        _tag(api, args.repo_id, args.tag)

    print(f"\nDone! https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
