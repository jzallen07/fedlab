"""Best-effort telemetry emitter used by runtime components."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx


class MonitorEmitter:
    """Send telemetry events to monitor API over HTTP."""

    def __init__(self, base_url: str | None, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout_s = timeout_s

    def emit_event(
        self,
        *,
        event_type: str,
        run_id: str,
        node_id: str,
        role: str,
        status: str,
        round_number: int | None = None,
        latency_ms: float | None = None,
        payload_bytes: int | None = None,
        metrics: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Emit one event to monitor API; errors are intentionally swallowed."""
        if not self.base_url:
            return
        payload: dict[str, Any] = {
            "event_id": str(uuid4()),
            "ts": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "round": round_number,
            "node_id": node_id,
            "role": role,
            "event_type": event_type,
            "status": status,
            "latency_ms": latency_ms,
            "payload_bytes": payload_bytes,
            "metrics": metrics,
            "details": details,
        }
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                client.post(f"{self.base_url}/events", json=payload)
        except Exception:
            return
