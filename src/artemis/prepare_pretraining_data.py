"""Stream, quality-filter, deduplicate, and provenance-tag a pinned Hugging Face corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")
TOKEN_RE = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})\b")
PHONE_LIKE_RE = re.compile(r"(?<![0-9])\+?[0-9](?:[0-9\s().-]{6,}[0-9])(?![0-9])")


def stable_bucket(value: str, buckets: int = 10_000) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) % buckets


def shuffle_stream(records: Iterable[Any], *, seed: int | None, buffer_size: int = 10_000):
    """Apply a reproducible bounded-buffer shuffle to a streaming dataset."""
    if buffer_size < 1:
        raise ValueError("shuffle buffer_size must be positive")
    if seed is None:
        return records
    shuffle = getattr(records, "shuffle", None)
    if not callable(shuffle):
        raise TypeError("seeded streaming shuffle requires a dataset with a shuffle() method")
    return shuffle(seed=seed, buffer_size=buffer_size)


def obvious_sensitive_content(text: str) -> bool:
    if any(pattern.search(text) for pattern in (EMAIL_RE, IPV4_RE, PRIVATE_KEY_RE, TOKEN_RE)):
        return True
    # Conservative phone-like pattern; false positives are possible, so every
    # rejected document is counted in the source manifest for audit.
    return any(
        9 <= sum(char.isdigit() for char in match.group()) <= 15
        for match in PHONE_LIKE_RE.finditer(text)
    )


def quality_rejection_reason(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    min_alpha_ratio: float,
) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if obvious_sensitive_content(text):
        return "possible_personal_or_secret_data"
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return "empty"
    alpha_numeric = sum(char.isalnum() for char in visible)
    if alpha_numeric / len(visible) < min_alpha_ratio:
        return "low_alphanumeric_ratio"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        repeated = len(lines) - len(set(lines))
        if repeated / len(lines) > 0.35:
            return "repetitive_lines"
    return None


def prepare_stream(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_config: str | None,
    dataset_split: str,
    license_id: str,
    license_review_note: str,
    output_dir: Path,
    max_documents: int = 20_000,
    validation_percent: float = 1.0,
    min_chars: int = 300,
    max_chars: int = 40_000,
    min_alpha_ratio: float = 0.35,
    text_field: str = "text",
    sampling_seed: int | None = None,
    shuffle_buffer_size: int = 10_000,
    required_language: str = "en",
    min_language_score: float = 0.70,
) -> dict[str, Any]:
    if len(dataset_revision) < 7 or dataset_revision in {"main", "master", "latest"}:
        raise ValueError("dataset_revision must be a pinned commit or immutable release")
    if not license_id.strip() or not license_review_note.strip():
        raise ValueError("license_id and an explicit license_review_note are required")
    if not 0 < validation_percent < 20:
        raise ValueError("validation_percent must be >0 and <20")
    if max_documents < 100:
        raise ValueError("max_documents must be at least 100")
    if shuffle_buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be positive")
    if not 0.0 <= min_language_score <= 1.0:
        raise ValueError("min_language_score must be between zero and one")
    if not isinstance(required_language, str) or required_language.strip().lower() != "en":
        raise ValueError("this training pipeline accepts English only (required_language='en')")
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("invalid character limits")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    train_text_path = output_dir / "train.txt"
    validation_text_path = output_dir / "validation.txt"
    provenance_path = output_dir / "provenance.jsonl"

    counts: Counter[str] = Counter()
    seen_content_hashes: set[str] = set()
    estimated_chars = 0
    accepted = 0

    with (
        train_path.open("w", encoding="utf-8") as train_jsonl,
        validation_path.open("w", encoding="utf-8") as validation_jsonl,
        train_text_path.open("w", encoding="utf-8") as train_text,
        validation_text_path.open("w", encoding="utf-8") as validation_text,
        provenance_path.open("w", encoding="utf-8") as provenance_stream,
    ):
        for record in records:
            if accepted >= max_documents:
                break
            language_label = record.get("language")
            normalized_language = (
                str(language_label).strip().lower() if language_label is not None else ""
            )
            if normalized_language not in {"en", "eng", "english"}:
                counts["rejected_language_filter"] += 1
                continue
            language_score = record.get("language_score")
            if (
                isinstance(language_score, bool)
                or not isinstance(language_score, (int, float))
                or not 0.0 <= float(language_score) <= 1.0
                or float(language_score) < min_language_score
            ):
                counts["rejected_language_confidence"] += 1
                continue
            raw_text = record.get(text_field)
            if not isinstance(raw_text, str):
                counts["rejected_missing_text"] += 1
                continue
            text = raw_text.strip()
            reason = quality_rejection_reason(
                text,
                min_chars=min_chars,
                max_chars=max_chars,
                min_alpha_ratio=min_alpha_ratio,
            )
            if reason is not None:
                counts[f"rejected_{reason}"] += 1
                continue

            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_hash in seen_content_hashes:
                counts["rejected_exact_duplicate"] += 1
                continue
            seen_content_hashes.add(content_hash)

            source_id = record.get("id")
            if not isinstance(source_id, (str, int)) or not str(source_id).strip():
                source_id = content_hash
            split_key = f"{dataset_id}:{dataset_revision}:{source_id}"
            split = (
                "validation"
                if stable_bucket(split_key) < int(validation_percent * 100)
                else "train"
            )

            metadata = {
                "dataset_id": dataset_id,
                "dataset_revision": dataset_revision,
                "dataset_config": dataset_config,
                "upstream_split": dataset_split,
                "source_id": str(source_id),
                "content_sha256": content_hash,
                "license": license_id,
                "split": split,
            }
            for field in ("url", "date", "dump", "language", "language_score", "token_count"):
                value = record.get(field)
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    metadata[field] = value

            example = {"text": text, "provenance": metadata}
            target_jsonl = validation_jsonl if split == "validation" else train_jsonl
            target_text = validation_text if split == "validation" else train_text
            target_jsonl.write(json.dumps(example, ensure_ascii=False) + "\n")
            target_text.write(text.replace("\r", "") + "\n\n")
            provenance_stream.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
            counts[f"accepted_{split}"] += 1
            accepted += 1
            estimated_chars += len(text)

    manifest = {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "dataset_config": dataset_config,
        "upstream_split": dataset_split,
        "license": license_id,
        "license_review_note": license_review_note,
        "max_documents": max_documents,
        "validation_percent": validation_percent,
        "language_policy": {
            "required_language": required_language,
            "accepted_labels": ["en", "eng", "english"],
            "minimum_language_score": min_language_score,
            "missing_language_metadata": "rejected",
        },
        "sampling": {
            "method": "huggingface_iterable_bounded_buffer_shuffle" if sampling_seed is not None else "upstream_stream_order",
            "seed": sampling_seed,
            "buffer_size": shuffle_buffer_size if sampling_seed is not None else None,
        },
        "quality_policy": {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_alpha_ratio": min_alpha_ratio,
            "filters": [
                "missing/non-string text",
                "obvious email/IP/private-key/token and phone-like number patterns",
                "length and low alphanumeric ratio",
                "highly repetitive lines",
                "exact duplicate content",
            ],
            "limitations": [
                "Regex filters are not comprehensive PII or secrets detection; phone-like filtering can create false positives.",
                "Exact deduplication does not detect semantic or near-duplicate documents.",
                "A dataset-level licence label does not replace upstream terms or component-level rights review.",
            ],
        },
        "counts": dict(counts),
        "accepted_documents": accepted,
        "accepted_characters": estimated_chars,
        "files": {
            "train_jsonl": "train.jsonl",
            "validation_jsonl": "validation.jsonl",
            "train_text_for_tokenizer": "train.txt",
            "validation_text_for_inspection": "validation.txt",
            "provenance_jsonl": "provenance.jsonl",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repository id")
    parser.add_argument("--config", help="pinned dataset configuration name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", required=True, help="immutable dataset commit SHA")
    parser.add_argument("--license", required=True, dest="license_id")
    parser.add_argument(
        "--license-review-note",
        required=True,
        help="human review note; this is recorded, not treated as legal advice",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=20_000)
    parser.add_argument("--validation-percent", type=float, default=1.0)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=40_000)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.35)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--shuffle-seed", type=int, help="deterministic per-cycle streaming shuffle seed")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--required-language",
        choices=("en",),
        default="en",
        help="strictly require a recognised English label and language confidence score",
    )
    parser.add_argument("--min-language-score", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the optional data dependencies: pip install -e '.[data]'") from exc

    stream = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    stream = shuffle_stream(stream, seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
    manifest = prepare_stream(
        stream,
        dataset_id=args.dataset,
        dataset_revision=args.revision,
        dataset_config=args.config,
        dataset_split=args.split,
        license_id=args.license_id,
        license_review_note=args.license_review_note,
        output_dir=args.output_dir,
        max_documents=args.max_documents,
        validation_percent=args.validation_percent,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        text_field=args.text_field,
        sampling_seed=args.shuffle_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        required_language=args.required_language,
        min_language_score=args.min_language_score,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
