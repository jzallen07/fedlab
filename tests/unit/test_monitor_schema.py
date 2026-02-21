from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.monitor.schema import TelemetryEvent


def test_telemetry_event_requires_run_id() -> None:
    with pytest.raises(ValidationError):
        TelemetryEvent(
            node_id="server",
            role="server",
            event_type="round_started",
            status="running",
        )


def test_telemetry_event_accepts_expected_payload() -> None:
    event = TelemetryEvent(
        run_id="run-1",
        round=2,
        node_id="client-1",
        role="client",
        event_type="client_train_completed",
        status="running",
        metrics={"loss": 0.3},
    )
    assert event.run_id == "run-1"
    assert event.metrics == {"loss": 0.3}
