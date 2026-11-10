from __future__ import annotations

import pytest

from artemis.prepare_s3_mix import merge_sft_sources, split_turn_documents


def turn(prompt: str, response: str) -> str:
    return (
        "<|prompter|>\n"
        + prompt
        + "\n<|assistant|>\n"
        + response
        + "\n<|end|>\n"
    )


def test_split_turn_documents_preserves_complete_sft_examples() -> None:
    source = turn("First question?", "First answer.") + "\n" + turn(
        "Second question?", "Second answer."
    )
    documents = split_turn_documents(source, source="fixture")
    assert len(documents) == 2
    assert all(document.endswith("<|end|>\n") for document in documents)
    assert "First answer." in documents[0]
    assert "Second answer." in documents[1]


def test_merge_deduplicates_and_prevents_train_validation_overlap() -> None:
    shared = turn("A shared prompt?", "A shared answer.")
    train_only = turn("A training prompt?", "A training answer.")
    validation_only = turn("A validation prompt?", "A validation answer.")

    train_text, validation_text, stats = merge_sft_sources(
        {
            "aya": shared + train_only,
            "oasst1": shared,
        },
        {
            "aya": validation_only,
            "oasst1": shared + validation_only,
        },
        seed=2031,
    )

    assert train_text.count("<|end|>") == 2
    assert validation_text.count("<|end|>") == 1
    assert "A shared prompt?" in train_text
    assert "A shared prompt?" not in validation_text
    assert stats["dropped_train_duplicates"] == 1
    assert stats["dropped_validation_train_overlap"] == 1
    assert stats["dropped_validation_duplicates"] == 0
    assert stats["sources"]["aya"]["train_documents"] == 2
    assert stats["sources"]["oasst1"]["train_documents"] == 0 or (
        stats["sources"]["oasst1"]["train_documents"] == 0
    )


def test_merge_requires_nonempty_validation_and_matching_source_names() -> None:
    sample = turn("Question?", "Answer.")
    with pytest.raises(ValueError, match="source names must match"):
        merge_sft_sources({"aya": sample}, {"oasst1": sample}, seed=1)
    with pytest.raises(ValueError, match="non-empty train and validation"):
        merge_sft_sources({"aya": sample}, {"aya": sample}, seed=1)


def test_split_rejects_records_without_assistant_markers() -> None:
    with pytest.raises(ValueError, match="chat control markers"):
        split_turn_documents("<|prompter|>\nNo assistant turn.", source="bad-fixture")
