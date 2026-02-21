"""FastAPI service exposing telemetry ingest, snapshots, and control APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from src.monitor.control import InvalidTransitionError, next_state
from src.monitor.schema import ControlRequest, RunAction, SnapshotResponse, TelemetryEvent
from src.monitor.store import MonitorStore
from src.monitor.ws import WSManager


def _event_name_for_action(action: RunAction) -> str:
    return {
        "start": "run_requested",
        "pause": "run_paused",
        "resume": "run_resumed",
        "stop": "run_stopped",
    }[action]


def create_app(database_path: Path | None = None) -> FastAPI:
    """Create and configure monitor FastAPI app."""
    db_path = database_path or Path("artifacts/monitor/monitor.db")
    store = MonitorStore.from_sqlite_path(db_path)
    ws_manager = WSManager()

    app = FastAPI(title="FedForge Monitor API", version="0.1.0")
    app.state.store = store
    app.state.ws = ws_manager

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/snapshot", response_model=SnapshotResponse)
    def snapshot(run_id: str = "local-run", limit: int = 200) -> SnapshotResponse:
        return store.snapshot(run_id=run_id, limit=limit)

    @app.get("/events", response_model=list[TelemetryEvent])
    def events(run_id: str = "local-run", limit: int = 200) -> list[TelemetryEvent]:
        return store.recent_events(run_id=run_id, limit=limit)

    @app.post("/events", response_model=TelemetryEvent)
    async def ingest_event(event: TelemetryEvent) -> TelemetryEvent:
        store.add_event(event)
        await ws_manager.broadcast({"type": "event", "payload": event.model_dump(mode="json")})
        return event

    @app.post("/control/{action}")
    async def control(action: RunAction, request: ControlRequest) -> dict[str, str]:
        current = store.get_run_state(request.run_id)
        try:
            new_state = next_state(current.state, action)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        state_snapshot = store.set_run_state(request.run_id, new_state)
        control_event = TelemetryEvent(
            run_id=request.run_id,
            node_id="operator",
            role="server",
            event_type=_event_name_for_action(action),
            status=new_state,
            details={"reason": request.reason} if request.reason else None,
        )
        store.add_event(control_event)
        await ws_manager.broadcast(
            {"type": "event", "payload": control_event.model_dump(mode="json")}
        )
        await ws_manager.broadcast(
            {"type": "run_state", "payload": state_snapshot.model_dump(mode="json")}
        )
        return {"run_id": state_snapshot.run_id, "state": state_snapshot.state}

    @app.websocket("/ws/events")
    async def ws_events(socket: WebSocket) -> None:
        await ws_manager.connect(socket)
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            await ws_manager.disconnect(socket)
        except Exception:
            await ws_manager.disconnect(socket)

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedForge monitor API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--db-path", type=Path, default=Path("artifacts/monitor/monitor.db"))
    return parser.parse_args(argv)


def main() -> None:
    """Run the monitor API with uvicorn."""
    import uvicorn

    args = parse_args()
    app = create_app(database_path=args.db_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
