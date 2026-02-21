"""Monitor service data access and snapshot composition."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import desc, select
from sqlalchemy.orm import sessionmaker

from src.monitor.db import (
    EventRecord,
    RunStateRecord,
    create_session_factory,
    sqlite_url_from_path,
)
from src.monitor.schema import RunStateSnapshot, SnapshotResponse, TelemetryEvent


class MonitorStore:
    """SQLite-backed monitor event and run-state store."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    @classmethod
    def from_sqlite_path(cls, path: Path) -> MonitorStore:
        session_factory = create_session_factory(sqlite_url_from_path(path))
        return cls(session_factory)

    def add_event(self, event: TelemetryEvent) -> None:
        """Persist one telemetry event; ignore duplicates by event_id."""
        with self._session_factory() as session:
            existing = session.scalar(
                select(EventRecord).where(EventRecord.event_id == event.event_id)
            )
            if existing is not None:
                return
            record = EventRecord(
                event_id=event.event_id,
                ts=event.ts,
                run_id=event.run_id,
                round=event.round,
                node_id=event.node_id,
                role=event.role,
                event_type=event.event_type,
                status=event.status,
                latency_ms=event.latency_ms,
                payload_bytes=event.payload_bytes,
                metrics_json=json.dumps(event.metrics) if event.metrics is not None else None,
                details_json=json.dumps(event.details) if event.details is not None else None,
            )
            session.add(record)
            session.commit()

    def set_run_state(self, run_id: str, state: str) -> RunStateSnapshot:
        """Set and persist run state."""
        now = datetime.now(UTC)
        with self._session_factory() as session:
            record = session.get(RunStateRecord, run_id)
            if record is None:
                record = RunStateRecord(run_id=run_id, state=state, updated_at=now)
                session.add(record)
            else:
                record.state = state
                record.updated_at = now
            session.commit()
            return RunStateSnapshot(
                run_id=record.run_id,
                state=record.state,
                updated_at=record.updated_at,
            )

    def get_run_state(self, run_id: str) -> RunStateSnapshot:
        """Get current run state or create an idle default."""
        with self._session_factory() as session:
            record = session.get(RunStateRecord, run_id)
            if record is None:
                now = datetime.now(UTC)
                record = RunStateRecord(run_id=run_id, state="idle", updated_at=now)
                session.add(record)
                session.commit()
            return RunStateSnapshot(
                run_id=record.run_id,
                state=record.state,
                updated_at=record.updated_at,
            )

    def recent_events(self, run_id: str, limit: int = 200) -> list[TelemetryEvent]:
        """Return recent events in ascending timestamp order."""
        with self._session_factory() as session:
            rows = session.scalars(
                select(EventRecord)
                .where(EventRecord.run_id == run_id)
                .order_by(desc(EventRecord.ts))
                .limit(limit)
            ).all()
        rows = list(reversed(rows))
        events: list[TelemetryEvent] = []
        for row in rows:
            events.append(
                TelemetryEvent(
                    event_id=row.event_id,
                    ts=row.ts,
                    run_id=row.run_id,
                    round=row.round,
                    node_id=row.node_id,
                    role=row.role,  # type: ignore[arg-type]
                    event_type=row.event_type,  # type: ignore[arg-type]
                    status=row.status,
                    latency_ms=row.latency_ms,
                    payload_bytes=row.payload_bytes,
                    metrics=json.loads(row.metrics_json) if row.metrics_json else None,
                    details=json.loads(row.details_json) if row.details_json else None,
                )
            )
        return events

    def snapshot(self, run_id: str, limit: int = 200) -> SnapshotResponse:
        """Return run state plus recent events."""
        return SnapshotResponse(
            run=self.get_run_state(run_id),
            recent_events=self.recent_events(run_id=run_id, limit=limit),
        )
