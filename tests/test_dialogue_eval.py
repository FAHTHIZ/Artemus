import json
from pathlib import Path

import pytest

from artemis.dialogue_eval import (
    build_dialogue_prompt,
    extract_assistant_response,
    load_scenarios,
)


class DummyTokenizer:
    stop_token_id = 7

    def encode(self, text):
        return list(text)

    def decode(self, ids):
        return "".join({1: "Hello", 2: " there", 7: "<|end|>"}[i] for i in ids)


def test_build_dialogue_prompt_preserves_turn_order():
    prompt = build_dialogue_prompt(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        "How are you?",
    )
    assert prompt == (
        "<|prompter|>\nHi\n<|end|>\n"
        "<|assistant|>\nHello\n<|end|>\n"
        "<|prompter|>\nHow are you?\n<|assistant|>\n"
    )


def test_build_dialogue_prompt_rejects_bad_history():
    with pytest.raises(ValueError, match="user/assistant"):
        build_dialogue_prompt([{"role": "system", "content": "bad"}], "Hi")
    with pytest.raises(ValueError, match="non-empty"):
        build_dialogue_prompt([], "  ")


def test_fit_dialogue_context_drops_oldest_complete_turns():
    from artemis.dialogue_eval import fit_dialogue_context

    history = [
        {"role": "user", "content": "old " * 30},
        {"role": "assistant", "content": "answer " * 30},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "kept"},
    ]
    prompt, token_ids, dropped = fit_dialogue_context(
        DummyTokenizer(), history, "Continue?", block_size=100
    )
    assert len(token_ids) <= 100
    assert dropped == 2
    assert "old " not in prompt
    assert "recent" in prompt
    assert "Continue?" in prompt


def test_fit_dialogue_context_rejects_an_overlong_current_turn():
    from artemis.dialogue_eval import fit_dialogue_context

    with pytest.raises(ValueError, match="current user turn"):
        fit_dialogue_context(DummyTokenizer(), [], "x" * 200, block_size=64)


def test_extract_assistant_response_detects_end_marker():
    text, ended = extract_assistant_response(DummyTokenizer(), [1, 2, 7])
    assert text == "Hello there"
    assert ended is True


def test_extract_assistant_response_handles_missing_end_marker():
    text, ended = extract_assistant_response(DummyTokenizer(), [1, 2])
    assert text == "Hello there"
    assert ended is False


def test_load_scenarios_validates_unique_ids(tmp_path: Path):
    path = tmp_path / "scenarios.jsonl"
    row = {"id": "a", "capability": "math", "turns": ["First?", "Follow-up?"]}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_scenarios(path)
