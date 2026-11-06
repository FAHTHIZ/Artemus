import json

import torch
import pytest

from artemis.token_cache import build_jsonl_token_cache
from artemis.tokenizer import ByteTokenizer


def test_jsonl_token_cache_adds_eos_per_document_and_reuses_verified_cache(tmp_path):
    source = tmp_path / "documents.jsonl"
    docs = [
        {"text": "alpha beta gamma " * 10, "id": "one"},
        {"text": "delta epsilon zeta " * 10, "id": "two"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in docs),
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    tokenizer = ByteTokenizer()
    first, first_metadata = build_jsonl_token_cache(
        source, tokenizer, cache_dir, text_field="text"
    )
    second, second_metadata = build_jsonl_token_cache(
        source, tokenizer, cache_dir, text_field="text"
    )

    expected = []
    for row in docs:
        expected.extend(tokenizer.encode(row["text"], add_eos=True))
    assert first.dtype == torch.int32
    assert first.tolist() == expected
    assert second.tolist() == expected
    assert first_metadata == second_metadata
    assert first_metadata["token_count"] == len(expected)
    assert len(first_metadata["cache_sha256"]) == 64


def test_jsonl_token_cache_rejects_invalid_rows(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps({"body": "missing text field"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain string field"):
        build_jsonl_token_cache(
            source, ByteTokenizer(), tmp_path / "cache", text_field="text"
        )


def test_jsonl_token_cache_invalidates_when_source_changes(tmp_path):
    source = tmp_path / "documents.jsonl"
    source.write_text(json.dumps({"text": "alpha beta gamma " * 10}) + "\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    tokenizer = ByteTokenizer()
    _, before = build_jsonl_token_cache(source, tokenizer, cache_dir)

    source.write_text(json.dumps({"text": "changed corpus content " * 10}) + "\n", encoding="utf-8")
    _, after = build_jsonl_token_cache(source, tokenizer, cache_dir)

    assert before["source_sha256"] != after["source_sha256"]
    assert before["cache_file"] != after["cache_file"]
