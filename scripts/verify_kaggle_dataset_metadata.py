"""Validate Kaggle dataset metadata downloaded by the Kaggle CLI.

Kaggle's metadata download may omit the dataset's API identity field. The
requested dataset identifier is already resolved by the successful
`kaggle datasets metadata <owner/slug>` command, so this helper treats the
downloaded `id` as optional while still checking the expected title and any
identity value that Kaggle does provide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_metadata(
    metadata: Any, expected_dataset_id: str, expected_title: str
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Kaggle dataset metadata must be a JSON object.")

    actual_id = metadata.get("id")
    if actual_id not in (None, "", expected_dataset_id):
        raise ValueError(
            f"Dataset metadata ID mismatch: expected {expected_dataset_id!r}, "
            f"got {actual_id!r}."
        )

    actual_title = metadata.get("title")
    if actual_title != expected_title:
        raise ValueError(
            f"Dataset metadata title mismatch: expected {expected_title!r}, "
            f"got {actual_title!r}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("expected_dataset_id")
    parser.add_argument("expected_title")
    args = parser.parse_args()

    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata, args.expected_dataset_id, args.expected_title)
    if metadata.get("id") in (None, ""):
        print(
            "Kaggle resolved the requested dataset; its downloaded metadata "
            "does not include an id field."
        )
    print("Verified Kaggle dataset:", args.expected_dataset_id)


if __name__ == "__main__":
    main()
