import hashlib
import json

import pytest

from artemis.prepare_aya import format_aya_example, prepare_aya_stream


def test_aya_format_uses_chat_markers_and_final_target():
    text = format_aya_example("What is photosynthesis?", "Plants use light to make sugars.")
    assert text == (
        "<|prompter|>\nWhat is photosynthesis?\n"
        "<|assistant|>\nPlants use light to make sugars.\n<|end|>"
    )


def test_aya_stream_preserves_language_and_discards_demographics(tmp_path):
    rows = [
        {
            "inputs": f"Explain a general concept number {index}.",
            "targets": f"A reviewed explanation for concept {index}.",
            "language": "English",
            "language_code": "eng",
            "annotation_type": "original-annotations",
            "user_id": f"private-user-{index}",
            "age_range": [25, 35],
        }
        for index in range(400)
    ]
    manifest = prepare_aya_stream(
        rows,
        dataset_revision="a" * 40,
        output_dir=tmp_path,
        max_examples=400,
        validation_percent=10,
    )

    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line)
        for line in (tmp_path / "validation.jsonl").read_text().splitlines()
    ]
    assert manifest["accepted_examples"] == 400
    assert train and validation
    assert manifest["examples_by_language_code"]["eng"] == 400
    assert all("user_id" not in row for row in train + validation)
    assert all("user_id" not in row["provenance"] for row in train + validation)
    assert "<|assistant|>" in train[0]["text"]
    assert manifest["demographic_fields_discarded"]


def test_aya_keeps_identical_prompts_in_same_split_and_deduplicates_pairs(tmp_path):
    rows = [
        {"inputs": "Same prompt?", "targets": "First answer.", "language_code": "eng"},
        {"inputs": "Same prompt?", "targets": "Second reviewed answer.", "language_code": "eng"},
        {"inputs": "Same prompt?", "targets": "First answer.", "language_code": "eng"},
    ]
    rows.extend(
        {
            "inputs": f"Unique prompt {index}?",
            "targets": f"Unique response {index}.",
            "language_code": "eng",
        }
        for index in range(300)
    )
    result = prepare_aya_stream(
        rows,
        dataset_revision="b" * 40,
        output_dir=tmp_path,
        max_examples=303,
        validation_percent=10,
    )
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line)
        for line in (tmp_path / "validation.jsonl").read_text().splitlines()
    ]
    same_prompt_sha = hashlib.sha256("Same prompt?".encode()).hexdigest()
    same_prompt_splits = {
        row["provenance"]["split"]
        for row in train + validation
        if row["provenance"]["prompt_sha256"] == same_prompt_sha
    }
    assert result["counts"]["rejected_exact_duplicate"] == 1
    assert len(same_prompt_splits) == 1


def test_aya_rejects_sensitive_or_unpinned_input(tmp_path):
    rows = [
        {
            "inputs": "Please contact alice@example.com with this question.",
            "targets": "I cannot process personal contact details.",
            "language_code": "eng",
        }
    ]
    with pytest.raises(ValueError, match="empty split"):
        prepare_aya_stream(
            rows,
            dataset_revision="c" * 40,
            output_dir=tmp_path,
            max_examples=100,
        )
    with pytest.raises(ValueError, match="pinned commit"):
        prepare_aya_stream(
            [],
            dataset_revision="main",
            output_dir=tmp_path / "other",
        )



def test_aya_manifest_records_deterministic_stream_sampling(tmp_path):
    rows = [
        {
            "inputs": f"Explain a unique safe concept number {index}.",
            "targets": f"A reviewed explanation for concept {index}.",
            "language_code": "eng",
        }
        for index in range(300)
    ]
    result = prepare_aya_stream(
        rows,
        dataset_revision="d" * 40,
        output_dir=tmp_path,
        max_examples=300,
        validation_percent=10,
        sampling_seed=2026,
        shuffle_buffer_size=512,
    )
    assert result["sampling"] == {
        "method": "huggingface_iterable_bounded_buffer_shuffle",
        "seed": 2026,
        "buffer_size": 512,
        "allowed_language_codes": ["eng"],
    }



def test_aya_language_allowlist_keeps_initial_sft_english_first(tmp_path):
    rows = [
        {
            "inputs": f"Explain one English research concept {index}.",
            "targets": f"This is a reviewed English explanation {index}.",
            "language": "English",
            "language_code": "eng",
        }
        for index in range(120)
    ] + [
        {
            "inputs": f"Explain this additional concept for an unapproved language label {index}.",
            "targets": f"This record uses a non-approved source label and must be excluded {index}.",
            "language": "Unapproved",
            "language_code": "xx",
        }
        for index in range(120)
    ]

    result = prepare_aya_stream(
        rows,
        dataset_revision="e" * 40,
        output_dir=tmp_path,
        max_examples=120,
        validation_percent=10,
        allowed_language_codes={"eng"},
    )
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line)
        for line in (tmp_path / "validation.jsonl").read_text().splitlines()
    ]
    assert result["accepted_examples"] == 120
    assert result["examples_by_language_code"] == {"eng": 120}
    assert result["counts"]["rejected_language_not_allowed"] == 120
    assert all(row["provenance"]["language_code"] == "eng" for row in train + validation)
    assert result["sampling"]["allowed_language_codes"] == ["eng"]
