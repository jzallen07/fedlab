from __future__ import annotations

from fastapi.testclient import TestClient

from src.monitor.app import create_app


def test_monitor_app_control_transitions_and_events() -> None:
    client = TestClient(create_app())

    run_id = "run-api-test"
    assert client.post("/control/start", json={"run_id": run_id}).status_code == 200
    assert client.post("/control/pause", json={"run_id": run_id}).status_code == 200
    assert client.post("/control/resume", json={"run_id": run_id}).status_code == 200
    assert client.post("/control/stop", json={"run_id": run_id}).status_code == 200

    snapshot = client.get("/snapshot", params={"run_id": run_id})
    assert snapshot.status_code == 200
    assert snapshot.json()["run"]["state"] == "stopped"

    events = client.get("/events", params={"run_id": run_id})
    assert events.status_code == 200
    assert len(events.json()) >= 4


def test_monitor_app_websocket_receives_streamed_events() -> None:
    client = TestClient(create_app())
    payload = {
        "run_id": "run-ws-test",
        "round": 1,
        "node_id": "client_1",
        "role": "client",
        "event_type": "client_train_started",
        "status": "running",
    }

    with client.websocket_connect("/ws/events") as ws:
        ws.send_text("subscribe")
        response = client.post("/events", json=payload)
        assert response.status_code == 200
        message = ws.receive_json()

    assert message["type"] == "event"
    assert message["payload"]["run_id"] == payload["run_id"]
    assert message["payload"]["event_type"] == payload["event_type"]
