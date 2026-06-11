"""Upload the LeetCode dataset to HuggingFace Hub (Stage 1, step 3).

Validates that the selected local Parquet files and the dataset card exist,
then pushes them as a single atomic commit to tkeskin/leetcode-solutions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from llm_fine_tune import loaders

REPO_ID = "tkeskin/leetcode-solutions"
REPO_TYPE = "dataset"

DATASET_CARD_PATH = Path("dataset_card/README.md")

_DATASET_FILES: dict[str, list[tuple[Path, str]]] = {
    "base": [
        (loaders.OUTPUT_DIR / "leetcode-solutions.parquet", "run `make base` first."),
    ],
    "instruct": [
        (
            loaders.OUTPUT_DIR / "leetcode-instruct-train.parquet",
            "run `make instruct` first.",
        ),
        (
            loaders.OUTPUT_DIR / "leetcode-instruct-test.parquet",
            "run `make instruct` first.",
        ),
    ],
    "evaluation": [
        (
            loaders.OUTPUT_DIR / "leetcode-evaluation.parquet",
            "run `make evaluation` first.",
        ),
    ],
}

DEFAULT_COMMIT_MESSAGE = "Update dataset"


def main() -> None:
    args = _parse_args()
    datasets = args.datasets or list(_DATASET_FILES)
    _require_files(datasets)
    api = HfApi()
    operations = _build_commit_operations(datasets)
    _commit_to_hub(api, operations, args.message)
    print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")


# ---- Argument parsing ----


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the leetcode-solutions dataset to HuggingFace."
    )
    parser.add_argument(
        "--datasets",
        action="append",
        choices=list(_DATASET_FILES),
        dest="datasets",
        metavar="DATASET",
        help="Dataset(s) to upload: base, instruct, evaluation (repeatable; default: all).",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message for the upload.",
    )
    return parser.parse_args()


# ---- Pre-flight checks ----


def _require_files(datasets: list[str]) -> None:
    loaders.require_file(DATASET_CARD_PATH, "dataset_card/README.md is missing.")
    for dataset in datasets:
        for path, hint in _DATASET_FILES[dataset]:
            loaders.require_file(path, hint)


# ---- Upload ----


def _build_commit_operations(datasets: list[str]) -> list[CommitOperationAdd]:
    ops = [
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=str(DATASET_CARD_PATH),
        )
    ]
    for dataset in datasets:
        for path, _ in _DATASET_FILES[dataset]:
            ops.append(
                CommitOperationAdd(
                    path_in_repo=path.name,
                    path_or_fileobj=str(path),
                )
            )
    return ops


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
