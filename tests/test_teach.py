import json

import pytest

from artemis.teach import format_sft_text, write_candidate


def test_format_sft_text_uses_chat_control_tokens():
    sample = format_sft_text("What is 2 + 2?", "4")
    assert sample == (
        "<|prompter|>\nWhat is 2 + 2?\n"
        "<|assistant|>\n4\n<|end|>"
    )


def test_human_correction_is_saved_with_provenance(tmp_path):
    path = tmp_path / "candidate.jsonl"
    write_candidate(
        path,
        prompt="Explain photosynthesis.",
        answer="Plants convert light energy into chemical energy.",
        session_id="session-1",
        turn=2,
        source_type="human_correction",
        checkpoint_sha256="a" * 64,
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["human_reviewed"] is True
    assert record["source_type"] == "human_correction"
    assert "<|assistant|>" in record["text"]
    assert record["checkpoint_sha256"] == "a" * 64


def test_unreviewed_or_unknown_source_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        write_candidate(
            tmp_path / "candidate.jsonl",
            prompt="Prompt",
            answer="Answer",
            session_id="session-1",
            turn=1,
            source_type="model_self_training",
            checkpoint_sha256="b" * 64,
        )


def test_format_sft_text_preserves_prior_multi_turn_context():
    sample = format_sft_text(
        "What did the second result mean?",
        "It was the validation result.",
        history=[
            {"role": "user", "content": "Compare these two results."},
            {"role": "assistant", "content": "The first is training; the second is validation."},
        ],
    )
    assert sample.count("<|prompter|>") == 1
    assert sample.count("<|assistant|>") == 1
    assert "Earlier assistant response (context only)" in sample
    assert "Compare these two results." in sample
    assert "What did the second result mean?" in sample


def test_format_sft_text_rejects_unknown_history_roles():
    with pytest.raises(ValueError, match="history messages"):
        format_sft_text("Prompt", "Answer", history=[{"role": "system", "content": "No"}])
