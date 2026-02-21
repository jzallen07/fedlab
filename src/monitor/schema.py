"""Telemetry and control schemas for monitor API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "node_heartbeat",
    "round_started",
    "model_dispatched",
    "client_train_started",
    "client_train_completed",
    "client_update_uploaded",
    "aggregation_started",
    "aggregation_completed",
    "round_completed",
    "node_error",
    "run_requested",
    "run_paused",
    "run_resumed",
    "run_stopped",
]
NodeRole = Literal["server", "client"]
RunState = Literal["idle", "running", "paused", "stopped", "error"]
RunAction = Literal["start", "pause", "resume", "stop"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class TelemetryEvent(BaseModel):
    """Event payload submitted by server/client runtimes."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    ts: datetime = Field(default_factory=utc_now)
    run_id: str
    round: int | None = None
    node_id: str
    role: NodeRole
    event_type: EventType
    status: str
    latency_ms: float | None = None
    payload_bytes: int | None = None
    metrics: dict[str, Any] | None = None
    details: dict[str, Any] | None = None


class ControlRequest(BaseModel):
    """Operator request for run lifecycle transitions."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = "local-run"
    reason: str | None = None


class RunStateSnapshot(BaseModel):
    """Current run state returned by snapshot and control APIs."""

    run_id: str
    state: RunState
    updated_at: datetime


class SnapshotResponse(BaseModel):
    """Monitor API snapshot response payload."""

    run: RunStateSnapshot
    recent_events: list[TelemetryEvent]
