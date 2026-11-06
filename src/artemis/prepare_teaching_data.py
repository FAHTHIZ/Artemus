"""Curate human-reviewed interactive teaching examples into leakage-aware SFT splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ALLOWED_SOURCE_TYPES = frozenset(
    {"human_correction", "human_approved_model_output"}
)


def session_split(session_id: str, validation_percent: int = 10) -> str:
    if not 1 <= validation_percent <= 30:
        raise ValueError("validation_percent must be between 1 and 30")
    bucket = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def curate_teaching_records(
    rows: list[Mapping[str, Any]],
    *,
    validation_percent: int = 10,
    minimum_text_chars: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Keep explicitly human-reviewed examples and split by whole teaching session."""
    if minimum_text_chars < 1:
        raise ValueError("minimum_text_chars must be positive")

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        if row.get("source") != "interactive_human_teaching":
            counts["rejected_wrong_source"] += 1
            continue
        if row.get("human_reviewed") is not True:
            counts["rejected_not_human_reviewed"] += 1
            continue
        source_type = row.get("source_type")
        if source_type not in ALLOWED_SOURCE_TYPES:
            counts["rejected_source_type"] += 1
            continue

        text = row.get("text")
        session_id = row.get("session_id")
        if not isinstance(text, str) or len(text.strip()) < minimum_text_chars:
            counts["rejected_text_too_short_or_missing"] += 1
            continue
        if not isinstance(session_id, str) or not session_id.strip():
            counts["rejected_missing_session_id"] += 1
            continue

        normalized_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if normalized_hash in seen:
            counts["rejected_duplicate"] += 1
            continue
        seen.add(normalized_hash)

        record = dict(row)
        record["text"] = text.strip()
        record["curation"] = {
            "input_row": index,
            "content_sha256": normalized_hash,
            "split": session_split(session_id, validation_percent),
        }
        if record["curation"]["split"] == "validation":
            validation.append(record)
            counts["accepted_validation"] += 1
        else:
            train.append(record)
            counts["accepted_train"] += 1

    if not train:
        raise ValueError(
            "No training examples remain. Add more reviewed sessions or adjust the split."
        )
    if not validation:
        raise ValueError(
            "No validation examples remain. Add more reviewed sessions or adjust validation_percent."
        )
    counts["accepted_total"] = len(train) + len(validation)
    return train, validation, dict(counts)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def prepare_teaching_data(
    input_path: Path,
    output_dir: Path,
    *,
    validation_percent: int = 10,
    minimum_text_chars: int = 40,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    train, validation, counts = curate_teaching_records(
        rows,
        validation_percent=validation_percent,
        minimum_text_chars=minimum_text_chars,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, examples in (("train", train), ("validation", validation)):
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for example in examples:
                stream.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "train.txt").open("w", encoding="utf-8") as stream:
        for example in train:
            stream.write(example["text"] + "\n")
    manifest = {
        "source": str(input_path),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "method": "human_reviewed_interactive_teaching",
        "validation_split": "stable hash of complete teaching session ID",
        "allowed_source_types": sorted(ALLOWED_SOURCE_TYPES),
        "counts": counts,
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "tokenizer_text": "train.txt",
        },
        "note": (
            "Candidate data is still subject to human review. No unreviewed model output "
            "is accepted by this curator. Validation sessions are never used for tokenizer fitting."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--minimum-text-chars", type=int, default=40)
    args = parser.parse_args()
    summary = prepare_teaching_data(
        args.input,
        args.output_dir,
        validation_percent=args.validation_percent,
        minimum_text_chars=args.minimum_text_chars,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
