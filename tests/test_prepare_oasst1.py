from artemis.prepare_oasst1 import build_examples, is_eligible


def message(message_id, parent_id, role, text, tree="tree-1", **overrides):
    row = {
        "message_id": message_id,
        "parent_id": parent_id,
        "message_tree_id": tree,
        "role": role,
        "text": text,
        "lang": "en",
        "review_result": True,
        "deleted": False,
        "synthetic": False,
        "labels": {},
    }
    row.update(overrides)
    return row


def test_oasst_filter_rejects_unreviewed_synthetic_and_pii():
    good = message("1", None, "prompter", "Explain recursion.")
    assert is_eligible(good)
    assert not is_eligible({**good, "review_result": False})
    assert not is_eligible({**good, "synthetic": True})
    assert not is_eligible({**good, "labels": {"pii": {"value": 0.4, "count": 3}}})


def test_build_examples_serializes_prompt_and_answer():
    rows = [
        message("1", None, "prompter", "What is a loop?"),
        message("2", "1", "assistant", "A loop repeats instructions."),
    ]
    examples, eligible = build_examples(rows, max_examples=20)
    assert eligible == 2
    assert len(examples) == 1
    _, text = examples[0]
    assert "<|prompter|>\nWhat is a loop?" in text
    assert "<|assistant|>\nA loop repeats instructions." in text
    assert text.endswith("<|end|>\n")


def test_build_examples_keeps_conversation_tree_contiguous():
    rows = [
        message("1", None, "prompter", "First question."),
        message("2", "1", "assistant", "First answer."),
        message("3", "2", "prompter", "Follow-up."),
        message("4", "3", "assistant", "Follow-up answer."),
    ]
    examples, _ = build_examples(rows, max_examples=20)
    texts = [text for _, text in examples]
    assert any("First question." in text and "Follow-up answer." in text for text in texts)
