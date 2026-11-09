"""Verify a prepared English-only pretraining source before bundling it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path, *, split: str, minimum_language_score: float) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Prepared {split} file is missing or empty: {path.name}")
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path.name}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path.name}:{line_number}")
            provenance: Any = row.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"Missing provenance in {path.name}:{line_number}")
            language = provenance.get("language")
            if not isinstance(language, str) or language.strip().lower() not in {"en", "eng", "english"}:
                raise ValueError(f"Non-English or missing language label in {path.name}:{line_number}")
            score = provenance.get("language_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"Missing language confidence in {path.name}:{line_number}")
            if not minimum_language_score <= float(score) <= 1.0:
                raise ValueError(f"Language confidence below threshold in {path.name}:{line_number}")
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Missing text in {path.name}:{line_number}")
            digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            if provenance.get("content_sha256") != digest:
                raise ValueError(f"Content hash mismatch in {path.name}:{line_number}")
            count += 1
    if count == 0:
        raise ValueError(f"Prepared {split} file contains no valid records.")
    return count


def verify_prepared_source(
    output_dir: Path,
    *,
    minimum_language_score: float = 0.70,
    expected_dataset_id: str | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be a JSON object.")
    if expected_dataset_id and manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError(
            f"Dataset ID mismatch: expected {expected_dataset_id!r}, "
            f"got {manifest.get('dataset_id')!r}."
        )
    if manifest.get("dataset_split") != "train" and manifest.get("upstream_split") != "train":
        raise ValueError("Only the upstream train split may be bundled for pretraining.")
    policy = manifest.get("language_policy")
    if not isinstance(policy, dict) or policy.get("required_language") != "en":
        raise ValueError("Source manifest does not declare the mandatory English-only language policy.")
    if float(policy.get("minimum_language_score", -1)) < minimum_language_score:
        raise ValueError("Source manifest language-confidence threshold is too low.")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Source manifest is missing its file map.")
    train_path = output_dir / str(files.get("train_jsonl", ""))
    validation_path = output_dir / str(files.get("validation_jsonl", ""))
    provenance_path = output_dir / str(files.get("provenance_jsonl", ""))
    train_count = _read_jsonl(train_path, split="train", minimum_language_score=minimum_language_score)
    validation_count = _read_jsonl(
        validation_path, split="validation", minimum_language_score=minimum_language_score
    )
    if not provenance_path.is_file() or provenance_path.stat().st_size == 0:
        raise ValueError("Source provenance JSONL is missing or empty.")
    with provenance_path.open("r", encoding="utf-8") as stream:
        provenance_count = sum(1 for line in stream if line.strip())
    expected_count = manifest.get("accepted_documents")
    if expected_count != train_count + validation_count or provenance_count != expected_count:
        raise ValueError(
            "Prepared split/provenance counts do not match the manifest: "
            f"train={train_count}, validation={validation_count}, "
            f"provenance={provenance_count}, manifest={expected_count}."
        )
    counts = manifest.get("counts", {})
    if counts.get("accepted_train") != train_count or counts.get("accepted_validation") != validation_count:
        raise ValueError("Accepted train/validation counts do not match the source manifest.")
    return {
        "dataset_id": manifest.get("dataset_id"),
        "dataset_revision": manifest.get("dataset_revision"),
        "accepted_documents": expected_count,
        "train_documents": train_count,
        "validation_documents": validation_count,
        "language_policy": policy,
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-language-score", type=float, default=0.70)
    parser.add_argument("--expected-dataset-id")
    args = parser.parse_args()
    result = verify_prepared_source(
        args.output_dir,
        minimum_language_score=args.minimum_language_score,
        expected_dataset_id=args.expected_dataset_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
