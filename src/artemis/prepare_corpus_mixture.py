"""Create deterministic, provenance-preserving document mixtures from curated JSONL splits.

Sampling quotas are document-based. Actual token proportions can vary with document
length and tokenizer fertility; measure them after fitting the final tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_documents(
    path: Path,
    *,
    text_field: str,
    excluded_hashes: set[str],
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Mixture source file does not exist: {path}")
    descriptors: list[dict[str, Any]] = []
    local_hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                raise ValueError(
                    f"JSONL row at {path}:{line_number} must contain string field {text_field!r}"
                )
            text = row[text_field].strip()
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in excluded_hashes or digest in local_hashes:
                continue
            local_hashes.add(digest)
            descriptors.append({
                "line_number": line_number,
                "content_sha256": digest,
                "character_count": len(text),
            })
    return descriptors


def _read_selected_rows(
    path: Path,
    selected: list[dict[str, Any]],
    *,
    text_field: str,
    source_id: str,
    weight: float,
    split: str,
) -> list[dict[str, Any]]:
    by_line = {int(item["line_number"]): item for item in selected}
    result: list[dict[str, Any]] = []
    if not by_line:
        return result
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            descriptor = by_line.get(line_number)
            if descriptor is None:
                continue
            row = json.loads(line)
            text = row[text_field].strip()
            record = dict(row)
            record[text_field] = text
            record["mixture"] = {
                "source_id": source_id,
                "source_weight": weight,
                "split": split,
                "content_sha256": descriptor["content_sha256"],
                "source_line_number": line_number,
            }
            result.append(record)
    return result


def _select_source(
    *,
    source_id: str,
    path: Path,
    quota: int,
    weight: float,
    text_field: str,
    seed: int,
    split: str,
    excluded_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    descriptors = _scan_documents(
        path, text_field=text_field, excluded_hashes=excluded_hashes
    )
    rng = random.Random(seed)
    rng.shuffle(descriptors)
    selected = descriptors[:quota]
    records = _read_selected_rows(
        path,
        selected,
        text_field=text_field,
        source_id=source_id,
        weight=weight,
        split=split,
    )
    hashes = {str(item["content_sha256"]) for item in selected}
    stats = {
        "source_id": source_id,
        "weight": weight,
        "input_file": str(path),
        "input_sha256": sha256_file(path),
        "eligible_unique_documents": len(descriptors),
        "selected_documents": len(records),
        "selected_characters": sum(len(row[text_field]) for row in records),
        "requested_document_quota": quota,
        "split": split,
    }
    return records, stats, hashes


def prepare_mixture(
    sources: list[dict[str, Any]],
    output_dir: Path,
    *,
    train_documents: int = 10_000,
    validation_documents: int = 500,
    text_field: str = "text",
    seed: int = 1337,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one source is required")
    if train_documents < 1 or validation_documents < 1:
        raise ValueError("train and validation document budgets must be positive")
    if not text_field.strip():
        raise ValueError("text_field must not be empty")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in sources:
        source_id = str(source.get("source_id", "")).strip()
        train_path = Path(source.get("train_path", ""))
        validation_path = Path(source.get("validation_path", ""))
        weight = float(source.get("weight", 0))
        if not source_id or source_id in seen_ids:
            raise ValueError("source IDs must be non-empty and unique")
        if not train_path.is_file() or not validation_path.is_file():
            raise FileNotFoundError(f"train/validation JSONL missing for source {source_id}")
        if weight <= 0:
            raise ValueError(f"source weight must be positive for {source_id}")
        seen_ids.add(source_id)
        normalized.append({
            "source_id": source_id,
            "train_path": train_path,
            "validation_path": validation_path,
            "weight": weight,
        })

    weight_total = sum(source["weight"] for source in normalized)
    train_seen: set[str] = set()
    validation_seen: set[str] = set()
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    train_stats: list[dict[str, Any]] = []
    validation_stats: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for index, source in enumerate(normalized):
        quota = max(1, round(train_documents * source["weight"] / weight_total))
        records, stats, hashes = _select_source(
            source_id=source["source_id"],
            path=source["train_path"],
            quota=quota,
            weight=source["weight"],
            text_field=text_field,
            seed=seed + index * 101,
            split="train",
            excluded_hashes=train_seen,
        )
        train_rows.extend(records)
        train_seen.update(hashes)
        train_stats.append(stats)

    # Validation is processed after training and excludes every selected train
    # document, preventing cross-source and train/validation content leakage.
    for index, source in enumerate(normalized):
        quota = max(1, round(validation_documents * source["weight"] / weight_total))
        records, stats, hashes = _select_source(
            source_id=source["source_id"],
            path=source["validation_path"],
            quota=quota,
            weight=source["weight"],
            text_field=text_field,
            seed=seed + 50_000 + index * 101,
            split="validation",
            excluded_hashes=train_seen | validation_seen,
        )
        validation_rows.extend(records)
        validation_seen.update(hashes)
        validation_stats.append(stats)

    if not train_rows or not validation_rows:
        raise ValueError(
            "Mixture produced an empty split; increase source samples or adjust quotas"
        )

    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_split(name: str, rows: list[dict[str, Any]]) -> None:
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as jsonl_stream, (
            output_dir / f"{name}.txt"
        ).open("w", encoding="utf-8") as text_stream:
            for row in rows:
                jsonl_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                text_stream.write(row[text_field] + "\n\n")

    write_split("train", train_rows)
    write_split("validation", validation_rows)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(row["mixture"]["source_id"] for row in rows)
        chars = Counter()
        for row in rows:
            chars[row["mixture"]["source_id"]] += len(row[text_field])
        return {
            "documents": len(rows),
            "characters": sum(chars.values()),
            "per_source_documents": dict(counts),
            "per_source_characters": dict(chars),
        }

    manifest = {
        "method": "deterministic_weighted_document_sampling",
        "seed": seed,
        "text_field": text_field,
        "document_weights_are_approximate_token_weights": True,
        "warning": (
            "Document quotas do not guarantee token quotas. Measure per-source token "
            "shares after fitting the actual tokenizer; adjust document weights if necessary."
        ),
        "sources": [
            {
                "source_id": source["source_id"],
                "weight": source["weight"],
                "train_path": str(source["train_path"]),
                "validation_path": str(source["validation_path"]),
            }
            for source in normalized
        ],
        "requested_budgets": {
            "train_documents": train_documents,
            "validation_documents": validation_documents,
        },
        "train_source_stats": train_stats,
        "validation_source_stats": validation_stats,
        "train": summarize(train_rows),
        "validation": summarize(validation_rows),
        "train_validation_hash_overlap": len(train_seen & validation_seen),
        "files": {
            "train_jsonl": "train.jsonl",
            "validation_jsonl": "validation.jsonl",
            "tokenizer_text": "train.txt",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs=4,
        action="append",
        metavar=("ID", "TRAIN_JSONL", "VALIDATION_JSONL", "WEIGHT"),
        required=True,
        help="repeat for each source; document weights are normalised automatically",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-documents", type=int, default=10_000)
    parser.add_argument("--validation-documents", type=int, default=500)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [
        {
            "source_id": source_id,
            "train_path": train_path,
            "validation_path": validation_path,
            "weight": float(weight),
        }
        for source_id, train_path, validation_path, weight in args.source
    ]
    manifest = prepare_mixture(
        sources,
        args.output_dir,
        train_documents=args.train_documents,
        validation_documents=args.validation_documents,
        text_field=args.text_field,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
