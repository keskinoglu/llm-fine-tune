from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "tkeskin/leetcode-solutions"
REPO_TYPE = "dataset"
BASE_PARQUET_PATH = Path("output/leetcode-solutions.parquet")
INSTRUCT_TRAIN_PATH = Path("output/leetcode-instruct-train.parquet")
INSTRUCT_TEST_PATH = Path("output/leetcode-instruct-test.parquet")
DATASET_CARD_PATH = Path("dataset_card/README.md")
DEFAULT_COMMIT_MESSAGE = "Update dataset"


def _ensure_files_exist() -> None:
    """Raise FileNotFoundError with actionable messages if required files are missing."""
    if not BASE_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{BASE_PARQUET_PATH} not found — run `make base` first."
        )
    if not INSTRUCT_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"{INSTRUCT_TRAIN_PATH} not found — run `make instruct` first."
        )
    if not INSTRUCT_TEST_PATH.exists():
        raise FileNotFoundError(
            f"{INSTRUCT_TEST_PATH} not found — run `make instruct` first."
        )
    if not DATASET_CARD_PATH.exists():
        raise FileNotFoundError(f"{DATASET_CARD_PATH} not found.")


def _build_commit_operations() -> list[CommitOperationAdd]:
    """Return the HF commit operations for the dataset card and all Parquet files."""
    return [
        CommitOperationAdd(
            path_in_repo="README.md", path_or_fileobj=str(DATASET_CARD_PATH)
        ),
        CommitOperationAdd(
            path_in_repo=BASE_PARQUET_PATH.name, path_or_fileobj=str(BASE_PARQUET_PATH)
        ),
        CommitOperationAdd(
            path_in_repo=INSTRUCT_TRAIN_PATH.name,
            path_or_fileobj=str(INSTRUCT_TRAIN_PATH),
        ),
        CommitOperationAdd(
            path_in_repo=INSTRUCT_TEST_PATH.name,
            path_or_fileobj=str(INSTRUCT_TEST_PATH),
        ),
    ]


def _commit_to_hub(
    api: HfApi, operations: list[CommitOperationAdd], message: str
) -> None:
    """Push all operations as a single atomic commit to the HF dataset repo."""
    print(f"Uploading {len(operations)} files to {REPO_ID} ...")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message=message,
    )


def main() -> None:
    """Entry point. Validates local files, then uploads them to HuggingFace."""
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
