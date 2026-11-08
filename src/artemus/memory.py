"""Hash-chained SQLite episodic event store for Artemus."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EpisodicEvent:
    sequence: int
    task_id: str
    event_type: str
    payload: Mapping[str, Any]
    created_at: str
    scope: str
    source: str
    previous_hash: str
    event_hash: str


class EpisodicMemory:
    """Persist append-only events with provenance and a tamper-evident hash chain.

    The hash chain detects accidental or unauthorised edits to stored event rows.
    It is not a digital signature: an attacker able to rewrite the database and
    recompute every hash can still replace the ledger.
    """

    def __init__(self, database: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._closed = False
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
        self._migrate_schema()
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_events_task "
            "ON episodic_events(task_id, sequence)"
        )
        self._connection.commit()

    def _migrate_schema(self) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(episodic_events)")
        }
        additions = {
            "created_at": "TEXT NOT NULL DEFAULT 'legacy'",
            "scope": "TEXT NOT NULL DEFAULT 'legacy'",
            "source": "TEXT NOT NULL DEFAULT 'migration'",
            "previous_hash": "TEXT NOT NULL DEFAULT ''",
            "event_hash": "TEXT NOT NULL DEFAULT ''",
        }
        for name, definition in additions.items():
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE episodic_events ADD COLUMN {name} {definition}"
                )

        # Backfill old databases once, in their original sequence order. The
        # historical wall-clock time is unknown and is deliberately not invented.
        rows = self._connection.execute(
            "SELECT * FROM episodic_events ORDER BY sequence"
        ).fetchall()
        previous_hash = ""
        for row in rows:
            if row["event_hash"]:
                previous_hash = str(row["event_hash"])
                continue
            fields = {
                "sequence": int(row["sequence"]),
                "task_id": str(row["task_id"]),
                "event_type": str(row["event_type"]),
                "payload_json": str(row["payload_json"]),
                "created_at": str(row["created_at"]),
                "scope": str(row["scope"]),
                "source": str(row["source"]),
                "previous_hash": previous_hash,
            }
            digest = self._digest(fields)
            self._connection.execute(
                "UPDATE episodic_events SET previous_hash = ?, event_hash = ? "
                "WHERE sequence = ?",
                (previous_hash, digest, int(row["sequence"])),
            )
            previous_hash = digest

    @staticmethod
    def _digest(fields: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            dict(fields), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> EpisodicEvent:
        return EpisodicEvent(
            sequence=int(row["sequence"]),
            task_id=str(row["task_id"]),
            event_type=str(row["event_type"]),
            payload=json.loads(row["payload_json"]),
            created_at=str(row["created_at"]),
            scope=str(row["scope"]),
            source=str(row["source"]),
            previous_hash=str(row["previous_hash"]),
            event_hash=str(row["event_hash"]),
        )

    def record(
        self,
        task_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        scope: str = "task",
        source: str = "kernel",
    ) -> EpisodicEvent:
        if self._closed:
            raise RuntimeError("episodic memory is closed")
        if not task_id.strip() or not event_type.strip():
            raise ValueError("task_id and event_type must not be empty")
        if not scope.strip() or not source.strip():
            raise ValueError("scope and source must not be empty")
        payload_json = json.dumps(
            dict(payload), sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._connection:
            previous = self._connection.execute(
                "SELECT event_hash FROM episodic_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = "" if previous is None else str(previous["event_hash"])
            cursor = self._connection.execute(
                """
                INSERT INTO episodic_events(
                    task_id, event_type, payload_json, created_at, scope, source,
                    previous_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '')
                """,
                (task_id, event_type, payload_json, created_at, scope, source, previous_hash),
            )
            sequence = int(cursor.lastrowid)
            fields = {
                "sequence": sequence,
                "task_id": task_id,
                "event_type": event_type,
                "payload_json": payload_json,
                "created_at": created_at,
                "scope": scope,
                "source": source,
                "previous_hash": previous_hash,
            }
            event_hash = self._digest(fields)
            self._connection.execute(
                "UPDATE episodic_events SET event_hash = ? WHERE sequence = ?",
                (event_hash, sequence),
            )
        return EpisodicEvent(
            sequence=sequence,
            task_id=task_id,
            event_type=event_type,
            payload=json.loads(payload_json),
            created_at=created_at,
            scope=scope,
            source=source,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )

    def events(self, task_id: str) -> list[EpisodicEvent]:
        if self._closed:
            raise RuntimeError("episodic memory is closed")
        rows = self._connection.execute(
            "SELECT * FROM episodic_events WHERE task_id = ? ORDER BY sequence",
            (task_id,),
        ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def verify_integrity(self) -> bool:
        """Verify every event and link in the global append-only sequence."""
        if self._closed:
            raise RuntimeError("episodic memory is closed")
        rows = self._connection.execute(
            "SELECT * FROM episodic_events ORDER BY sequence"
        ).fetchall()
        previous_hash = ""
        for row in rows:
            fields = {
                "sequence": int(row["sequence"]),
                "task_id": str(row["task_id"]),
                "event_type": str(row["event_type"]),
                "payload_json": str(row["payload_json"]),
                "created_at": str(row["created_at"]),
                "scope": str(row["scope"]),
                "source": str(row["source"]),
                "previous_hash": str(row["previous_hash"]),
            }
            if fields["previous_hash"] != previous_hash:
                return False
            expected = self._digest(fields)
            if str(row["event_hash"]) != expected:
                return False
            previous_hash = expected
        return True

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
