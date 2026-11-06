"""Small append-only SQLite episodic event store for Artemus."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EpisodicEvent:
    sequence: int
    task_id: str
    event_type: str
    payload: Mapping[str, Any]


class EpisodicMemory:
    """Persist task events without silently rewriting earlier observations."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_events_task "
            "ON episodic_events(task_id, sequence)"
        )
        self._connection.commit()
        self._closed = False

    def record(
        self,
        task_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EpisodicEvent:
        if self._closed:
            raise RuntimeError("episodic memory is closed")
        if not task_id.strip() or not event_type.strip():
            raise ValueError("task_id and event_type must not be empty")
        payload_json = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
        cursor = self._connection.execute(
            "INSERT INTO episodic_events(task_id, event_type, payload_json) "
            "VALUES (?, ?, ?)",
            (task_id, event_type, payload_json),
        )
        self._connection.commit()
        return EpisodicEvent(
            sequence=int(cursor.lastrowid),
            task_id=task_id,
            event_type=event_type,
            payload=json.loads(payload_json),
        )

    def events(self, task_id: str) -> list[EpisodicEvent]:
        if self._closed:
            raise RuntimeError("episodic memory is closed")
        rows = self._connection.execute(
            "SELECT sequence, task_id, event_type, payload_json "
            "FROM episodic_events WHERE task_id = ? ORDER BY sequence",
            (task_id,),
        ).fetchall()
        return [
            EpisodicEvent(
                sequence=int(row["sequence"]),
                task_id=str(row["task_id"]),
                event_type=str(row["event_type"]),
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> EpisodicMemory:
        if self._closed:
            raise RuntimeError("episodic memory is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
