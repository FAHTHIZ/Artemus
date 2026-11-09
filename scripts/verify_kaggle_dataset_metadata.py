"""Validate (and normalise) Kaggle dataset metadata downloaded by the Kaggle CLI.

Kaggle CLI 2.x writes ``dataset-metadata.json`` as the whole API response,
``{"info": {"datasetSlug": ..., "ownerUser": ..., "title": ...}}``, because it
calls ``response.to_json(response.info)``. Older CLIs wrote a flat object with
``id`` and ``title``. This helper accepts both shapes, may find the dataset
identity in ``id`` or in ``ownerUser``/``datasetSlug``, checks the expected
title, and rewrites the file as flat metadata with an explicit ``id`` so that
``kaggle datasets version -p`` can reuse it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def unwrap_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Kaggle dataset metadata must be a JSON object.")
    info = metadata.get("info")
    if "title" not in metadata and isinstance(info, dict):
        return dict(info)
    return dict(metadata)


def resolved_identity(metadata: dict[str, Any]) -> str | None:
    actual_id = metadata.get("id")
    if actual_id not in (None, ""):
        return str(actual_id)
    owner = metadata.get("ownerUser") or metadata.get("owner_user")
    slug = metadata.get("datasetSlug") or metadata.get("dataset_slug")
    if owner and slug:
        return f"{owner}/{slug}"
    return None


def validate_metadata(
    metadata: Any, expected_dataset_id: str, expected_title: str
) -> dict[str, Any]:
    flat = unwrap_metadata(metadata)

    actual_id = resolved_identity(flat)
    if actual_id is not None and actual_id.lower() != expected_dataset_id.lower():
        raise ValueError(
            f"Dataset metadata ID mismatch: expected {expected_dataset_id!r}, "
            f"got {actual_id!r}."
        )

    actual_title = flat.get("title")
    if actual_title != expected_title:
        raise ValueError(
            f"Dataset metadata title mismatch: expected {expected_title!r}, "
            f"got {actual_title!r}."
        )
    return flat


def normalized_metadata(flat: dict[str, Any], expected_dataset_id: str) -> dict[str, Any]:
    keep = ("title", "subtitle", "description", "keywords", "licenses", "resources")
    out: dict[str, Any] = {"id": expected_dataset_id}
    for key in keep:
        if flat.get(key) not in (None, "", []):
            out[key] = flat[key]
    if "licenses" not in out:
        out["licenses"] = [{"name": "other"}]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("expected_dataset_id")
    parser.add_argument("expected_title")
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="validate only; do not rewrite the file as flat metadata",
    )
    args = parser.parse_args()

    raw = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    flat = validate_metadata(raw, args.expected_dataset_id, args.expected_title)
    if resolved_identity(flat) is None:
        print(
            "Kaggle resolved the requested dataset; its downloaded metadata "
            "does not include an id field."
        )
    if not args.no_rewrite:
        args.metadata_path.write_text(
            json.dumps(normalized_metadata(flat, args.expected_dataset_id), indent=2) + "\n",
            encoding="utf-8",
        )
    print("Verified Kaggle dataset:", args.expected_dataset_id)


if __name__ == "__main__":
    main()
