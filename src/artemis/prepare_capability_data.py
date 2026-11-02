"""Build a reproducible multi-task SFT mixture from licensed train splits.

Inputs:
- OASST1 preprocessed train/validation conversation text.
- GSM8K official train JSONL only; 10% held out deterministically for validation.
- MBPP full JSONL; official task-ID ranges define train and validation.

GSM8K test and MBPP test/prompt IDs are not ingested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

GSM8K_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    + GSM8K_REVISION
    + "/grade_school_math/data/train.jsonl"
)
MBPP_GIT_COMMIT = "83f1e7b59e762648c953de751d31f75720683622"
MBPP_GIT_BLOB_SHA = "96e2dc2f7dd95b7b87b38dc46e42cbed444251bb"
MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    + MBPP_GIT_COMMIT
    + "/mbpp/mbpp.jsonl"
)
MBPP_LICENSE = "CC-BY-4.0"
GSM8K_LICENSE = "MIT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


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


def _hash_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 10


def build_gsm8k_examples(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Split only the official GSM8K train split; never read the official test file."""
    train: list[str] = []
    validation: list[str] = []
    seen: set[str] = set()
    for row in rows:
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            continue
        key = hashlib.sha256((question + "\0" + answer).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        example = (
            "<|prompter|>\nSolve this mathematics problem. Show a concise, "
            "checkable derivation and put the final answer on the last line.\n"
            + question
            + "\n<|assistant|>\n"
            + answer
            + "\n<|end|>\n"
        )
        (validation if _hash_bucket(key) == 0 else train).append(example)
    return train, validation


def build_mbpp_examples(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, int]]:
    """Use MBPP official train IDs 601-974 and validation IDs 511-600 only."""
    train: list[str] = []
    validation: list[str] = []
    counts = {"train": 0, "validation": 0, "excluded": 0}
    seen: set[int] = set()

    for row in rows:
        try:
            task_id = int(row.get("task_id"))
        except (TypeError, ValueError):
            counts["excluded"] += 1
            continue
        if task_id in seen:
            raise ValueError(f"Duplicate MBPP task_id: {task_id}")
        seen.add(task_id)

        if 601 <= task_id <= 974:
            split = train
            split_name = "train"
        elif 511 <= task_id <= 600:
            split = validation
            split_name = "validation"
        else:
            # IDs 1-10 are prompting examples; 11-510 are test; never ingest them.
            counts["excluded"] += 1
            continue

        task_text = row.get("text")
        solution = row.get("code")
        tests = row.get("test_list") or []
        setup = row.get("test_setup_code") or ""
        if (
            not isinstance(task_text, str)
            or not isinstance(solution, str)
            or not task_text.strip()
            or not solution.strip()
            or not isinstance(tests, list)
            or not all(isinstance(test, str) for test in tests)
        ):
            counts["excluded"] += 1
            continue

        test_block = "\n".join(([setup.strip()] if setup.strip() else []) + tests)
        prompt = (
            "Implement the requested Python task using the standard library. "
            "Return the complete solution code, not an explanation.\n"
            "Task: "
            + task_text.strip()
            + "\nProvided checks:\n"
            + test_block
        )
        split.append(
            "<|prompter|>\n"
            + prompt
            + "\n<|assistant|>\n"
            + solution.strip()
            + "\n<|end|>\n"
        )
        counts[split_name] += 1

    return train, validation, counts


def prepare_mixture(
    *,
    oasst_train: Path,
    oasst_validation: Path,
    gsm8k_train: Path,
    mbpp_full: Path,
    output_dir: Path,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if verify_hashes:
        gsm_hash = sha256_file(gsm8k_train)
        if gsm_hash != GSM8K_SHA256:
            raise ValueError(
                f"GSM8K SHA-256 mismatch: expected {GSM8K_SHA256}, got {gsm_hash}"
            )
        mbpp_blob = git_blob_sha(mbpp_full)
        if mbpp_blob != MBPP_GIT_BLOB_SHA:
            raise ValueError(
                f"MBPP Git blob SHA mismatch: expected {MBPP_GIT_BLOB_SHA}, got {mbpp_blob}"
            )
    else:
        gsm_hash = sha256_file(gsm8k_train)
        mbpp_blob = git_blob_sha(mbpp_full)

    gsm_train, gsm_validation = build_gsm8k_examples(load_jsonl(gsm8k_train))
    mbpp_train, mbpp_validation, mbpp_counts = build_mbpp_examples(load_jsonl(mbpp_full))

    oasst_train_text = oasst_train.read_text(encoding="utf-8").strip()
    oasst_validation_text = oasst_validation.read_text(encoding="utf-8").strip()
    if not oasst_train_text or not oasst_validation_text:
        raise ValueError("OASST1 train and validation inputs must both be non-empty")

    training_parts = [oasst_train_text, *gsm_train, *mbpp_train]
    validation_parts = [oasst_validation_text, *gsm_validation, *mbpp_validation]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.txt"
    validation_path = output_dir / "validation.txt"
    train_path.write_text("\n".join(training_parts) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(validation_parts) + "\n", encoding="utf-8")

    manifest = {
        "mixture": "OASST1 + GSM8K-main-train + MBPP-official-train-ids",
        "sources": {
            "oasst1": {
                "license": "Apache-2.0",
                "preprocessing": "reviewed English conversations; conversation-tree grouped split",
            },
            "gsm8k": {
                "license": GSM8K_LICENSE,
                "revision": GSM8K_REVISION,
                "url": GSM8K_URL,
                "train_source_sha256": gsm_hash,
                "official_train_examples": 7473,
                "validation": "deterministic 10% hash split from official train only",
                "official_test": "not downloaded or ingested",
            },
            "mbpp": {
                "license": MBPP_LICENSE,
                "revision": MBPP_GIT_COMMIT,
                "url": MBPP_URL,
                "git_blob_sha": mbpp_blob,
                "training_task_ids": "601-974",
                "validation_task_ids": "511-600",
                "excluded_task_ids": "1-10 prompt; 11-510 protected test",
                "counts": mbpp_counts,
            },
        },
        "mixture_policy": "concatenate each eligible source once; no benchmark test split in training",
        "train_components": {
            "oasst1": True,
            "gsm8k": len(gsm_train),
            "mbpp": len(mbpp_train),
        },
        "validation_components": {
            "oasst1": True,
            "gsm8k": len(gsm_validation),
            "mbpp": len(mbpp_validation),
        },
        "train_bytes": train_path.stat().st_size,
        "validation_bytes": validation_path.stat().st_size,
        "test_integrity": "GSM8K official test file and MBPP IDs 11-510 were not included",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasst-train", type=Path, required=True)
    parser.add_argument("--oasst-validation", type=Path, required=True)
    parser.add_argument("--gsm8k-train", type=Path, required=True)
    parser.add_argument("--mbpp-full", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mixture"))
    args = parser.parse_args()
    manifest = prepare_mixture(
        oasst_train=args.oasst_train,
        oasst_validation=args.oasst_validation,
        gsm8k_train=args.gsm8k_train,
        mbpp_full=args.mbpp_full,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
