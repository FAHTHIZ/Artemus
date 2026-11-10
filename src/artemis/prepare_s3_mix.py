"""Build the audited English-only joint S3 instruction-tuning mixture.

This stage intentionally consumes only the pinned train sources and each
source's own validation partition. Diagnostic examples are written separately
and never enter the optimizer data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Mapping

from artemis.prepare_aya import prepare_aya_stream
from artemis.prepare_capability_data import (
    GSM8K_LICENSE,
    GSM8K_REVISION,
    GSM8K_SHA256,
    GSM8K_URL,
    MBPP_GIT_BLOB_SHA,
    MBPP_GIT_COMMIT,
    MBPP_LICENSE,
    MBPP_URL,
    build_evaluation_records,
    build_gsm8k_examples,
    build_mbpp_examples,
    git_blob_sha,
    load_jsonl,
    sha256_file,
)
from artemis.prepare_oasst1 import (
    EXPECTED_SHA256 as OASST1_SHA256,
    SOURCE_URL as OASST1_URL,
    SOURCE_REVISION as OASST1_REVISION,
    prepare as prepare_oasst1,
)
from artemis.prepare_pretraining_data import shuffle_stream

AYA_DATASET = "CohereLabs/aya_dataset"
AYA_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    """Download a pinned public source and verify it before exposing the file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return destination

    temporary = destination.with_name(destination.name + ".download")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Artemus-Research-Data-Preparation/1.0"},
    )
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def split_turn_documents(text: str, *, source: str) -> list[str]:
    """Recover complete prompt/response documents from prepared SFT text."""
    documents: list[str] = []
    for fragment in text.replace("\r\n", "\n").replace("\r", "\n").split("<|end|>"):
        document = fragment.strip()
        if not document:
            continue
        if "<|prompter|>" not in document or "<|assistant|>" not in document:
            raise ValueError(
                f"{source} contains a record without both chat control markers"
            )
        documents.append(document + "\n<|end|>\n")
    if not documents:
        raise ValueError(f"{source} produced no prompt/response documents")
    return documents


def _document_sha256(document: str) -> str:
    normalized = "\n".join(line.rstrip() for line in document.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def merge_sft_sources(
    train_sources: Mapping[str, str],
    validation_sources: Mapping[str, str],
    *,
    seed: int,
) -> tuple[str, str, dict]:
    """Deduplicate across sources/splits, then deterministically shuffle documents."""
    if set(train_sources) != set(validation_sources):
        raise ValueError("Training and validation source names must match")

    seen_train: set[str] = set()
    seen_validation: set[str] = set()
    train_documents: list[str] = []
    validation_documents: list[str] = []
    per_source: dict[str, dict[str, int]] = {}
    dropped_train_duplicates = 0
    dropped_validation_duplicates = 0
    dropped_validation_train_overlap = 0

    for source, content in train_sources.items():
        source_documents = split_turn_documents(content, source=f"{source} train")
        accepted = 0
        for document in source_documents:
            digest = _document_sha256(document)
            if digest in seen_train:
                dropped_train_duplicates += 1
                continue
            seen_train.add(digest)
            train_documents.append(document)
            accepted += 1
        per_source[source] = {
            "train_documents": accepted,
            "validation_documents": 0,
        }
        if accepted == 0:
            raise ValueError(f"{source} has no unique training documents")

    for source, content in validation_sources.items():
        source_documents = split_turn_documents(content, source=f"{source} validation")
        accepted = 0
        for document in source_documents:
            digest = _document_sha256(document)
            if digest in seen_train:
                dropped_validation_train_overlap += 1
                continue
            if digest in seen_validation:
                dropped_validation_duplicates += 1
                continue
            seen_validation.add(digest)
            validation_documents.append(document)
            accepted += 1
        per_source[source]["validation_documents"] = accepted

    if not train_documents or not validation_documents:
        raise ValueError("The joint S3 mixture must have non-empty train and validation splits")
    if seen_train.intersection(seen_validation):
        raise AssertionError("A document crossed the S3 train/validation boundary")

    random.Random(seed).shuffle(train_documents)
    random.Random(seed + 1).shuffle(validation_documents)
    train_text = "\n".join(document.rstrip() for document in train_documents) + "\n"
    validation_text = "\n".join(
        document.rstrip() for document in validation_documents
    ) + "\n"

    stats = {
        "sources": per_source,
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "dropped_train_duplicates": dropped_train_duplicates,
        "dropped_validation_duplicates": dropped_validation_duplicates,
        "dropped_validation_train_overlap": dropped_validation_train_overlap,
        "seed": seed,
        "split_policy": (
            "Preserve upstream source splits; remove exact normalized document "
            "duplicates across all sources; shuffle documents deterministically."
        ),
    }
    return train_text, validation_text, stats


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def prepare_s3_mix(
    *,
    output_dir: Path,
    cache_dir: Path,
    seed: int = 2031,
    max_aya_examples: int = 10_000,
    max_oasst_examples: int = 15_000,
) -> dict:
    """Download pinned sources, prepare individual partitions, and build S3."""
    if max_aya_examples < 100:
        raise ValueError("max_aya_examples must be at least 100")
    if max_oasst_examples < 20:
        raise ValueError("max_oasst_examples must be at least 20")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Aya is fetched as a bounded stream; only records explicitly marked English
    # (language_code=eng) are admitted by the existing audited adapter.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install the optional data dependencies with: pip install -e '.[data]'"
        ) from exc

    aya_stream = load_dataset(
        AYA_DATASET,
        split="train",
        revision=AYA_REVISION,
        streaming=True,
    )
    aya_stream = shuffle_stream(aya_stream, seed=seed, buffer_size=10_000)
    aya_dir = output_dir / "sources" / "aya"
    aya_manifest = prepare_aya_stream(
        aya_stream,
        dataset_revision=AYA_REVISION,
        output_dir=aya_dir,
        max_examples=max_aya_examples,
        validation_percent=2.0,
        sampling_seed=seed,
        shuffle_buffer_size=10_000,
        allowed_language_codes={"eng"},
    )

    # OASST1 source bytes are pinned by SHA-256. Its adapter groups whole
    # conversation trees so sibling turns cannot leak across partitions.
    oasst_path = download_verified(
        OASST1_URL,
        cache_dir / "oasst1.messages.jsonl.gz",
        OASST1_SHA256,
    )
    oasst_dir = output_dir / "sources" / "oasst1"
    oasst_manifest = prepare_oasst1(
        oasst_path,
        oasst_dir,
        expected_sha256=OASST1_SHA256,
        max_examples=max_oasst_examples,
    )

    # GSM8K and MBPP are verified against their pinned upstream revisions and
    # use the repository's train/optimizer-validation/diagnostic partitioning.
    gsm_path = download_verified(
        GSM8K_URL,
        cache_dir / "gsm8k-train.jsonl",
        GSM8K_SHA256,
    )
    mbpp_path = cache_dir / "mbpp.jsonl"
    if not mbpp_path.is_file() or git_blob_sha(mbpp_path) != MBPP_GIT_BLOB_SHA:
        mbpp_path.unlink(missing_ok=True)
        request = urllib.request.Request(
            MBPP_URL,
            headers={"User-Agent": "Artemus-Research-Data-Preparation/1.0"},
        )
        temporary = mbpp_path.with_name(mbpp_path.name + ".download")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                temporary.write_bytes(response.read())
            if git_blob_sha(temporary) != MBPP_GIT_BLOB_SHA:
                raise ValueError("MBPP Git blob SHA does not match its pinned revision")
            temporary.replace(mbpp_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    gsm_rows = load_jsonl(gsm_path)
    mbpp_rows = load_jsonl(mbpp_path)
    gsm_train, gsm_validation = build_gsm8k_examples(gsm_rows)
    mbpp_train, mbpp_validation, mbpp_counts = build_mbpp_examples(mbpp_rows)
    evaluation_records = build_evaluation_records(gsm_rows, mbpp_rows)

    gsm_dir = output_dir / "sources" / "gsm8k"
    mbpp_dir = output_dir / "sources" / "mbpp"
    gsm_train_text = "\n".join(gsm_train)
    gsm_validation_text = "\n".join(gsm_validation)
    mbpp_train_text = "\n".join(mbpp_train)
    mbpp_validation_text = "\n".join(mbpp_validation)
    _write_text(gsm_dir / "train.txt", gsm_train_text)
    _write_text(gsm_dir / "validation.txt", gsm_validation_text)
    _write_text(mbpp_dir / "train.txt", mbpp_train_text)
    _write_text(mbpp_dir / "validation.txt", mbpp_validation_text)

    evaluation_path = output_dir / "evaluation.jsonl"
    with evaluation_path.open("w", encoding="utf-8") as stream:
        for record in evaluation_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_manifests = {
        "aya": aya_manifest,
        "oasst1": oasst_manifest,
        "gsm8k": {
            "dataset": "openai/grade-school-math",
            "revision": GSM8K_REVISION,
            "license": GSM8K_LICENSE,
            "train_source_sha256": sha256_file(gsm_path),
            "official_test_downloaded": False,
            "examples": {
                "train": len(gsm_train),
                "validation": len(gsm_validation),
                "diagnostic": sum(
                    row["task_type"] == "math" for row in evaluation_records
                ),
            },
        },
        "mbpp": {
            "dataset": "google-research/mbpp",
            "revision": MBPP_GIT_COMMIT,
            "license": MBPP_LICENSE,
            "git_blob_sha": git_blob_sha(mbpp_path),
            "task_id_policy": {
                "train": "601-974",
                "validation": "511-555",
                "diagnostic": "556-600",
                "excluded": "1-10 prompt examples; 11-510 protected test tasks",
            },
            "counts": mbpp_counts,
        },
    }

    train_sources = {
        "aya": (aya_dir / "train.txt").read_text(encoding="utf-8"),
        "oasst1": (oasst_dir / "train.txt").read_text(encoding="utf-8"),
        "gsm8k": gsm_train_text,
        "mbpp": mbpp_train_text,
    }
    validation_sources = {
        "aya": (aya_dir / "validation.txt").read_text(encoding="utf-8"),
        "oasst1": (oasst_dir / "validation.txt").read_text(encoding="utf-8"),
        "gsm8k": gsm_validation_text,
        "mbpp": mbpp_validation_text,
    }
    train_text, validation_text, mix_stats = merge_sft_sources(
        train_sources,
        validation_sources,
        seed=seed,
    )

    train_path = output_dir / "train.txt"
    validation_path = output_dir / "validation.txt"
    _write_text(train_path, train_text)
    _write_text(validation_path, validation_text)

    manifest = {
        "schema_version": 1,
        "stage": "S3 joint English instruction SFT",
        "parent_checkpoint": {
            "branch": "checkpoints",
            "path": "f2-edu2-6m-step17500/artemus-f2-edu2-6m-step17500.pt",
            "sha256": "c4a791f9941ef4488c74a51bf2a46fce40b0764aa8bd81683c35385939efd584",
            "global_step": 17_500,
            "parameters": 6_451_200,
        },
        "sources": source_manifests,
        "mixture": mix_stats,
        "files": {
            "train": train_path.name,
            "validation": validation_path.name,
            "diagnostics": evaluation_path.name,
        },
        "sha256": {
            "train": sha256_file(train_path),
            "validation": sha256_file(validation_path),
            "diagnostics": sha256_file(evaluation_path),
        },
        "bytes": {
            "train": train_path.stat().st_size,
            "validation": validation_path.stat().st_size,
            "diagnostics": evaluation_path.stat().st_size,
        },
        "data_policy": (
            "English only. Aya uses language_code=eng; OASST1 uses reviewed, "
            "non-deleted, non-synthetic English messages; GSM8K and MBPP use "
            "the repository's documented train/validation/diagnostic partitions. "
            "No official GSM8K test data or protected MBPP task IDs enter training."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/s3-mix"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/s3-source-cache"))
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--max-aya-examples", type=int, default=10_000)
    parser.add_argument("--max-oasst-examples", type=int, default=15_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_s3_mix(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        max_aya_examples=args.max_aya_examples,
        max_oasst_examples=args.max_oasst_examples,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
