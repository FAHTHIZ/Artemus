import sqlite3

import pytest

from artemus.memory import EpisodicMemory


def test_events_have_provenance_and_a_valid_hash_chain(tmp_path):
    database = tmp_path / "episodes.sqlite"
    with EpisodicMemory(database) as memory:
        first = memory.record(
            "task-a", "observation", {"value": 1}, scope="workspace", source="sensor:test"
        )
        second = memory.record("task-b", "action", {"ok": True})

        assert first.previous_hash == ""
        assert second.previous_hash == first.event_hash
        assert first.scope == "workspace"
        assert first.source == "sensor:test"
        assert first.created_at.endswith("+00:00")
        assert memory.verify_integrity()

    with EpisodicMemory(database) as reopened:
        assert reopened.verify_integrity()
        assert reopened.events("task-a")[0].event_hash == first.event_hash


def test_hash_chain_detects_payload_tampering(tmp_path):
    database = tmp_path / "episodes.sqlite"
    with EpisodicMemory(database) as memory:
        memory.record("task-a", "observation", {"value": "original"})

    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE episodic_events SET payload_json = ? WHERE task_id = ?",
        ('{"value":"changed"}', "task-a"),
    )
    connection.commit()
    connection.close()

    with EpisodicMemory(database) as reopened:
        assert not reopened.verify_integrity()


def test_legacy_ledger_is_migrated_without_inventing_timestamps(tmp_path):
    database = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE episodic_events ("
        "sequence INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, "
        "event_type TEXT NOT NULL, payload_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO episodic_events(task_id, event_type, payload_json) VALUES (?, ?, ?)",
        ("old-task", "observation", '{"state":"old"}'),
    )
    connection.commit()
    connection.close()

    with EpisodicMemory(database) as memory:
        events = memory.events("old-task")
        assert len(events) == 1
        assert events[0].created_at == "legacy"
        assert events[0].scope == "legacy"
        assert events[0].source == "migration"
        assert memory.verify_integrity()
        memory.record("new-task", "goal_created", {"goal": "new"})
        assert memory.verify_integrity()


def test_record_rejects_non_finite_json_numbers():
    with EpisodicMemory() as memory:
        with pytest.raises(ValueError):
            memory.record("task", "bad", {"value": float("nan")})
