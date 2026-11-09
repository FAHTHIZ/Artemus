"""Prepare human-annotated English Aya prompt/response pairs for SFT.

Only prompt/response text and non-sensitive source labels are retained. Annotator
IDs and demographic fields are intentionally discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from artemis.prepare_pretraining_data import obvious_sensitive_content, shuffle_stream, stable_bucket


def format_aya_example(prompt: str, response: str) -> str:
    if not prompt.strip() or not response.strip():
        raise ValueError("Aya prompt and response must not be empty")
    return (
        "<|prompter|>\n"
        + prompt.strip()
        + "\n<|assistant|>\n"
        + response.strip()
        + "\n<|end|>"
    )


def prepare_aya_stream(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_revision: str,
    output_dir: Path,
    max_examples: int = 10_000,
    validation_percent: float = 2.0,
    max_prompt_chars: int = 20_000,
    max_response_chars: int = 20_000,
    sampling_seed: int | None = None,
    shuffle_buffer_size: int = 10_000,
    allowed_language_codes: set[str] | None = None,
) -> dict[str, Any]:
    if len(dataset_revision) < 7 or dataset_revision in {"main", "master", "latest"}:
        raise ValueError("dataset_revision must be a pinned commit or immutable release")
    allowed_language_codes = {"eng"} if allowed_language_codes is None else set(allowed_language_codes)
    if allowed_language_codes != {"eng"}:
        raise ValueError("Aya preparation accepts English only (language code eng).")
    if max_examples < 100:
        raise ValueError("max_examples must be at least 100")
    if shuffle_buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be positive")
    if not 0 < validation_percent < 20:
        raise ValueError("validation_percent must be >0 and <20")
    if max_prompt_chars < 1 or max_response_chars < 1:
        raise ValueError("prompt and response character limits must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    train_text_path = output_dir / "train.txt"
    validation_text_path = output_dir / "validation.txt"
    provenance_path = output_dir / "provenance.jsonl"

    counts: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_annotation_type: Counter[str] = Counter()
    seen_pairs: set[str] = set()
    accepted = 0

    with (
        train_path.open("w", encoding="utf-8") as train_jsonl,
        validation_path.open("w", encoding="utf-8") as validation_jsonl,
        train_text_path.open("w", encoding="utf-8") as train_text,
        validation_text_path.open("w", encoding="utf-8") as validation_text,
        provenance_path.open("w", encoding="utf-8") as provenance_stream,
    ):
        for row_number, record in enumerate(records, start=1):
            if accepted >= max_examples:
                break
            raw_language_code = record.get("language_code")
            language_code = raw_language_code if isinstance(raw_language_code, str) else "und"
            if allowed_language_codes is not None and language_code not in allowed_language_codes:
                counts["rejected_language_not_allowed"] += 1
                continue

            prompt = record.get("inputs")
            response = record.get("targets")
            if not isinstance(prompt, str) or not isinstance(response, str):
                counts["rejected_missing_prompt_or_response"] += 1
                continue

            prompt = prompt.strip()
            response = response.strip()
            if not prompt or not response:
                counts["rejected_empty_prompt_or_response"] += 1
                continue
            if len(prompt) > max_prompt_chars or len(response) > max_response_chars:
                counts["rejected_too_long"] += 1
                continue
            if obvious_sensitive_content(prompt) or obvious_sensitive_content(response):
                counts["rejected_possible_personal_or_secret_data"] += 1
                continue

            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            pair_sha = hashlib.sha256(
                (prompt + "\0" + response).encode("utf-8")
            ).hexdigest()
            if pair_sha in seen_pairs:
                counts["rejected_exact_duplicate"] += 1
                continue
            seen_pairs.add(pair_sha)

            language = record.get("language")
            annotation_type = record.get("annotation_type")
            language = language if isinstance(language, str) else "unknown"
            language_code = language_code if isinstance(language_code, str) else "und"
            annotation_type = (
                annotation_type if isinstance(annotation_type, str) else "unspecified"
            )

            # Group all responses to an identical prompt in the same split.
            split_key = f"{dataset_revision}:{language_code}:{prompt_sha}"
            split = (
                "validation"
                if stable_bucket(split_key) < int(validation_percent * 100)
                else "train"
            )
            provenance = {
                "dataset_id": "CohereLabs/aya_dataset",
                "dataset_revision": dataset_revision,
                "dataset_license": "Apache-2.0",
                "source_row_number": row_number,
                "language": language,
                "language_code": language_code,
                "annotation_type": annotation_type,
                "prompt_sha256": prompt_sha,
                "pair_sha256": pair_sha,
                "split": split,
                "demographic_fields_discarded": True,
            }
            example = {"text": format_aya_example(prompt, response), "provenance": provenance}
            target_jsonl = validation_jsonl if split == "validation" else train_jsonl
            target_text = validation_text if split == "validation" else train_text
            target_jsonl.write(json.dumps(example, ensure_ascii=False) + "\n")
            target_text.write(example["text"].replace("\r", "") + "\n\n")
            provenance_stream.write(
                json.dumps(provenance, ensure_ascii=False, sort_keys=True) + "\n"
            )
            counts[f"accepted_{split}"] += 1
            by_language[language_code] += 1
            by_annotation_type[annotation_type] += 1
            accepted += 1

    if counts["accepted_train"] == 0 or counts["accepted_validation"] == 0:
        raise ValueError(
            "Aya preparation produced an empty split; increase max_examples or adjust validation_percent"
        )

    manifest = {
        "dataset_id": "CohereLabs/aya_dataset",
        "dataset_revision": dataset_revision,
        "license": "Apache-2.0",
        "license_review_note": (
            "The upstream card declares Apache-2.0. This adapter retains only input/target "
            "text and non-sensitive language/annotation labels; annotator IDs and demographic "
            "fields are discarded. Human annotation is not a guarantee of factual correctness."
        ),
        "split_policy": (
            "Deterministic split by dataset revision, language code, and prompt hash so "
            "re-annotations of an identical prompt stay in one split."
        ),
        "quality_policy": {
            "max_prompt_chars": max_prompt_chars,
            "max_response_chars": max_response_chars,
            "filters": [
                "missing or empty prompt/response",
                "overlong prompt or response",
                "obvious email/IP/private-key/token and phone-like patterns",
                "exact prompt-response duplicates",
            ],
            "limitations": [
                "Regex filters are not comprehensive PII or secrets detection.",
                "Human annotations may still contain factual or reasoning errors.",
                "Prompt-group splits do not guarantee semantic deduplication.",
            ],
        },
        "demographic_fields_discarded": [
            "user_id",
            "age_range",
            "gender",
            "languages",
            "dialects",
        ],
        "sampling": {
            "method": "huggingface_iterable_bounded_buffer_shuffle" if sampling_seed is not None else "upstream_stream_order",
            "seed": sampling_seed,
            "buffer_size": shuffle_buffer_size if sampling_seed is not None else None,
            "allowed_language_codes": sorted(allowed_language_codes) if allowed_language_codes is not None else None,
        },
        "accepted_examples": accepted,
        "counts": dict(counts),
        "examples_by_language_code": dict(by_language),
        "examples_by_annotation_type": dict(by_annotation_type),
        "files": {
            "train_jsonl": "train.jsonl",
            "validation_jsonl": "validation.jsonl",
            "train_text": "train.txt",
            "validation_text": "validation.txt",
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
    parser.add_argument("--dataset", default="CohereLabs/aya_dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", required=True, help="immutable dataset commit SHA")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--validation-percent", type=float, default=2.0)
    parser.add_argument("--max-prompt-chars", type=int, default=20_000)
    parser.add_argument("--max-response-chars", type=int, default=20_000)
    parser.add_argument("--shuffle-seed", type=int, help="deterministic per-cycle streaming shuffle seed")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--language-codes",
        nargs="+",
        choices=("eng",),
        default=["eng"],
        help="Aya preparation is restricted to English (eng).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the optional data dependencies: pip install -e '.[data]'") from exc
    if args.dataset != "CohereLabs/aya_dataset":
        raise SystemExit(
            "This adapter is intentionally scoped to the reviewed CohereLabs/aya_dataset source"
        )

    stream = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    stream = shuffle_stream(stream, seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
    manifest = prepare_aya_stream(
        stream,
        dataset_revision=args.revision,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        validation_percent=args.validation_percent,
        max_prompt_chars=args.max_prompt_chars,
        max_response_chars=args.max_response_chars,
        sampling_seed=args.shuffle_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        allowed_language_codes=set(args.language_codes),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
