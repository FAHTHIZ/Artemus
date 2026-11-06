import json

import pytest

from artemis.prepare_pretraining_data import prepare_stream, quality_rejection_reason


def test_quality_filter_rejects_email_and_private_key_patterns():
    assert quality_rejection_reason(
        "This is a sufficiently long document " * 20 + "contact me at alice@example.com",
        min_chars=20,
        max_chars=10000,
        min_alpha_ratio=0.3,
    ) == "possible_personal_or_secret_data"
    assert quality_rejection_reason(
        "This is a sufficiently long document " * 20 + "-----BEGIN PRIVATE KEY-----",
        min_chars=20,
        max_chars=10000,
        min_alpha_ratio=0.3,
    ) == "possible_personal_or_secret_data"


def test_prepare_stream_writes_provenance_and_deterministic_split(tmp_path):
    records = [
        {"id": f"doc-{index}", "text": (f"Document {index}. " + "Useful words, facts and explanation. " * 20)}
        for index in range(300)
    ]
    manifest = prepare_stream(
        records,
        dataset_id="example/corpus",
        dataset_revision="a" * 40,
        dataset_config="default",
        dataset_split="train",
        license_id="MIT",
        license_review_note="test-only review record",
        output_dir=tmp_path,
        max_documents=300,
        validation_percent=10,
        min_chars=100,
        max_chars=5000,
        min_alpha_ratio=0.3,
    )

    train_rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    validation_rows = [
        json.loads(line) for line in (tmp_path / "validation.jsonl").read_text().splitlines()
    ]
    assert manifest["accepted_documents"] == 300
    assert train_rows and validation_rows
    assert train_rows[0]["provenance"]["dataset_revision"] == "a" * 40
    assert train_rows[0]["provenance"]["license"] == "MIT"
    assert (tmp_path / "train.txt").is_file()
    assert (tmp_path / "manifest.json").is_file()


def test_prepare_stream_requires_immutable_revision_and_license_review(tmp_path):
    with pytest.raises(ValueError, match="pinned commit"):
        prepare_stream(
            [],
            dataset_id="example/corpus",
            dataset_revision="main",
            dataset_config=None,
            dataset_split="train",
            license_id="MIT",
            license_review_note="review",
            output_dir=tmp_path,
        )
