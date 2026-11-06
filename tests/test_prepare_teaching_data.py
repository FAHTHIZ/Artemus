import json

import pytest

from artemis.prepare_teaching_data import curate_teaching_records, prepare_teaching_data


def candidate(session_id: str, turn: int, text: str) -> dict:
    return {
        "text": text,
        "source": "interactive_human_teaching",
        "source_type": "human_correction",
        "human_reviewed": True,
        "session_id": session_id,
        "turn": turn,
        "checkpoint_sha256": "a" * 64,
    }


def test_teaching_split_keeps_whole_sessions_together() -> None:
    rows = []
    for session_num in range(100):
        session = f"session-{session_num}"
        rows.extend([
            candidate(session, 1, f"Prompt one and reviewed answer for session {session_num}."),
            candidate(session, 2, f"Prompt two and reviewed answer for session {session_num}."),
        ])
    train, validation, counts = curate_teaching_records(rows, validation_percent=10)
    train_sessions = {row["session_id"] for row in train}
    validation_sessions = {row["session_id"] for row in validation}
    assert train_sessions.isdisjoint(validation_sessions)
    assert train and validation
    assert counts["accepted_total"] == len(train) + len(validation)


def test_unreviewed_model_output_is_rejected() -> None:
    rows = [
        {
            **candidate("session-a", 1, "A sufficiently long candidate answer from a model."),
            "human_reviewed": False,
        },
        {
            **candidate("session-b", 1, "A second sufficiently long candidate answer."),
            "source_type": "model_self_training",
        },
    ]
    with pytest.raises(ValueError, match="No training examples"):
        curate_teaching_records(rows)


def test_prepare_teaching_data_writes_manifest_and_jsonl(tmp_path) -> None:
    source = tmp_path / "candidates.jsonl"
    rows = []
    for index in range(120):
        rows.append(candidate(
            f"session-{index}",
            1,
            f"Human-corrected prompt and detailed answer for session number {index}.",
        ))
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    manifest = prepare_teaching_data(source, output)
    assert manifest["counts"]["accepted_total"] == 120
    assert (output / "train.jsonl").is_file()
    assert (output / "validation.jsonl").is_file()
    assert (output / "manifest.json").is_file()
