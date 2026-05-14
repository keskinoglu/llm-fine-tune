from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "tkeskin/leetcode-solutions"
REPO_TYPE = "dataset"
PARQUET_PATH = Path("output/leetcode-solutions.parquet")
DATASET_CARD_PATH = Path("dataset_card/README.md")
DEFAULT_COMMIT_MESSAGE = "Update dataset"


def _ensure_files_exist() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found — run `make dataset` first.")
    if not DATASET_CARD_PATH.exists():
        raise FileNotFoundError(f"{DATASET_CARD_PATH} not found.")


def _build_commit_operations() -> list[CommitOperationAdd]:
    return [
        CommitOperationAdd(
            path_in_repo="README.md", path_or_fileobj=str(DATASET_CARD_PATH)
        ),
        CommitOperationAdd(
            path_in_repo=PARQUET_PATH.name, path_or_fileobj=str(PARQUET_PATH)
        ),
    ]


def _commit_to_hub(
    api: HfApi, operations: list[CommitOperationAdd], message: str
) -> None:
    print(f"Uploading {len(operations)} files to {REPO_ID} ...")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message=message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the leetcode-solutions dataset to HuggingFace."
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message for the upload.",
    )
    args = parser.parse_args()

    _ensure_files_exist()
    api = HfApi()
    operations = _build_commit_operations()
    _commit_to_hub(api, operations, args.message)
    print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
