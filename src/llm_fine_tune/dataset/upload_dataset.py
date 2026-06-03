"""Upload the LeetCode dataset to HuggingFace Hub (Stage 1, step 3).

Validates that all local Parquet files and the dataset card exist, then
pushes them as a single atomic commit to tkeskin/leetcode-solutions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from llm_fine_tune import loaders

REPO_ID = "tkeskin/leetcode-solutions"
REPO_TYPE = "dataset"

BASE_PARQUET_PATH = loaders.OUTPUT_DIR / "leetcode-solutions.parquet"
INSTRUCT_TRAIN_PATH = loaders.OUTPUT_DIR / "leetcode-instruct-train.parquet"
INSTRUCT_TEST_PATH = loaders.OUTPUT_DIR / "leetcode-instruct-test.parquet"
DATASET_CARD_PATH = Path("dataset_card/README.md")

DEFAULT_COMMIT_MESSAGE = "Update dataset"


def main() -> None:
    args = _parse_args()
    _require_all_files()
    api = HfApi()
    operations = _build_commit_operations()
    _commit_to_hub(api, operations, args.message)
    print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")


# ---- Argument parsing ----


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the leetcode-solutions dataset to HuggingFace."
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message for the upload.",
    )
    return parser.parse_args()


# ---- Pre-flight checks ----


def _require_all_files() -> None:
    loaders.require_file(BASE_PARQUET_PATH, "run `make base` first.")
    loaders.require_file(INSTRUCT_TRAIN_PATH, "run `make instruct` first.")
    loaders.require_file(INSTRUCT_TEST_PATH, "run `make instruct` first.")
    loaders.require_file(DATASET_CARD_PATH, "dataset_card/README.md is missing.")


# ---- Upload ----


def _build_commit_operations() -> list[CommitOperationAdd]:
    return [
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=str(DATASET_CARD_PATH),
        ),
        CommitOperationAdd(
            path_in_repo=BASE_PARQUET_PATH.name,
            path_or_fileobj=str(BASE_PARQUET_PATH),
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
    api: HfApi,
    operations: list[CommitOperationAdd],
    message: str,
) -> None:
    print(f"Uploading {len(operations)} files to {REPO_ID} ...")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message=message,
    )


if __name__ == "__main__":
    main()
