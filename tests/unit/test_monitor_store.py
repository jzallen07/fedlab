from __future__ import annotations

from pathlib import Path

from src.monitor.schema import TelemetryEvent
from src.monitor.store import MonitorStore


def test_monitor_store_persists_events_and_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.db"
    store = MonitorStore.from_sqlite_path(db_path)

    store.set_run_state("run-1", "running")
    store.add_event(
        TelemetryEvent(
            run_id="run-1",
            round=1,
            node_id="server",
            role="server",
            event_type="round_started",
            status="running",
            metrics={"selected_clients": 3},
        )
    )

    snapshot = store.snapshot(run_id="run-1", limit=10)
    assert snapshot.run.run_id == "run-1"
    assert snapshot.run.state == "running"
    assert len(snapshot.recent_events) == 1
    assert snapshot.recent_events[0].event_type == "round_started"
