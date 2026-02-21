"""Docker e2e smoke runner for realtime telemetry and control actions."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import websockets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run docker e2e smoke test for FedForge stack")
    parser.add_argument("--compose-file", type=Path, default=Path("docker-compose.yml"))
    parser.add_argument("--project-name", default="fedforge-smoke")
    parser.add_argument("--monitor-url", default="http://127.0.0.1:8090")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:5173")
    parser.add_argument("--training-run-id", default="docker-run")
    parser.add_argument("--control-run-id", default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument("--event-timeout-s", type=float, default=300.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    return parser.parse_args(argv)


def _compose_cmd(*, compose_file: Path, project_name: str, args: list[str]) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "-p",
        project_name,
        *args,
    ]


def _run_compose(*, compose_file: Path, project_name: str, args: list[str]) -> None:
    command = _compose_cmd(compose_file=compose_file, project_name=project_name, args=args)
    subprocess.run(command, check=True)


def _monitor_ws_url(monitor_url: str) -> str:
    if monitor_url.startswith("https://"):
        return f"wss://{monitor_url.removeprefix('https://').rstrip('/')}/ws/events"
    if monitor_url.startswith("http://"):
        return f"ws://{monitor_url.removeprefix('http://').rstrip('/')}/ws/events"
    raise ValueError("monitor-url must start with http:// or https://")


def _parse_event_ts(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _wait_for_condition(
    *,
    description: str,
    timeout_s: float,
    poll_interval_s: float,
    check: callable,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(poll_interval_s)
    if last_error is not None:
        raise TimeoutError(f"Timed out waiting for {description}: {last_error}") from last_error
    raise TimeoutError(f"Timed out waiting for {description}")


def _wait_for_http_health(
    *,
    client: httpx.Client,
    url: str,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    def _check() -> bool:
        response = client.get(url)
        return response.status_code == 200

    _wait_for_condition(
        description=f"HTTP health at {url}",
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        check=_check,
    )


def _wait_for_training_events(
    *,
    client: httpx.Client,
    monitor_url: str,
    run_id: str,
    started_at: datetime,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    required = {"round_started", "client_train_started"}

    def _check() -> bool:
        response = client.get(
            f"{monitor_url.rstrip('/')}/events",
            params={"run_id": run_id, "limit": 500},
        )
        response.raise_for_status()
        events = response.json()
        recent_types = {
            event["event_type"]
            for event in events
            if _parse_event_ts(str(event["ts"])) >= started_at
        }
        return required.issubset(recent_types)

    _wait_for_condition(
        description=f"training telemetry events for run_id={run_id}",
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        check=_check,
    )


async def _assert_ws_broadcast(
    *,
    monitor_url: str,
    timeout_s: float,
    event_payload: dict[str, object],
) -> None:
    ws_url = _monitor_ws_url(monitor_url)
    async with websockets.connect(ws_url) as socket:
        await socket.send("smoke-listener")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{monitor_url.rstrip('/')}/events",
                json=event_payload,
            )
            response.raise_for_status()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            message = await asyncio.wait_for(socket.recv(), timeout=max(remaining, 0.1))
            payload = json.loads(message)
            if payload.get("type") != "event":
                continue
            event = payload.get("payload", {})
            if event.get("event_id") == event_payload["event_id"]:
                return

    raise TimeoutError("Timed out waiting for websocket event broadcast")


def _assert_control_sequence(
    *,
    client: httpx.Client,
    monitor_url: str,
    run_id: str,
) -> None:
    sequence = [
        ("start", "running"),
        ("pause", "paused"),
        ("resume", "running"),
        ("stop", "stopped"),
    ]
    for action, expected_state in sequence:
        response = client.post(
            f"{monitor_url.rstrip('/')}/control/{action}",
            json={"run_id": run_id, "reason": "docker-smoke"},
        )
        response.raise_for_status()
        payload = response.json()
        state = payload.get("state")
        if state != expected_state:
            raise RuntimeError(
                f"control/{action} returned state={state!r}, expected {expected_state!r}"
            )

    snapshot = client.get(
        f"{monitor_url.rstrip('/')}/snapshot",
        params={"run_id": run_id, "limit": 50},
    )
    snapshot.raise_for_status()
    payload = snapshot.json()
    if payload.get("run", {}).get("state") != "stopped":
        raise RuntimeError("Control sequence did not end in stopped run state")

    event_types = [event["event_type"] for event in payload.get("recent_events", [])]
    required_events = {"run_requested", "run_paused", "run_resumed", "run_stopped"}
    if not required_events.issubset(set(event_types)):
        raise RuntimeError(
            "Control event audit missing expected lifecycle events: "
            f"required={sorted(required_events)} observed={sorted(set(event_types))}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    control_run_id = args.control_run_id or f"docker-smoke-{int(time.time())}"
    cleanup = not args.no_cleanup
    started_at = datetime.now(UTC)

    up_args = ["up", "-d"]
    if not args.skip_build:
        up_args.append("--build")

    _run_compose(
        compose_file=args.compose_file,
        project_name=args.project_name,
        args=up_args,
    )

    try:
        with httpx.Client(timeout=5.0) as client:
            _wait_for_http_health(
                client=client,
                url=f"{args.monitor_url.rstrip('/')}/health",
                timeout_s=args.startup_timeout_s,
                poll_interval_s=args.poll_interval_s,
            )
            _wait_for_http_health(
                client=client,
                url=f"{args.dashboard_url.rstrip('/')}/monitor/health",
                timeout_s=args.startup_timeout_s,
                poll_interval_s=args.poll_interval_s,
            )
            _wait_for_training_events(
                client=client,
                monitor_url=args.monitor_url,
                run_id=args.training_run_id,
                started_at=started_at,
                timeout_s=args.event_timeout_s,
                poll_interval_s=args.poll_interval_s,
            )

            ws_event_payload = {
                "event_id": str(uuid4()),
                "run_id": control_run_id,
                "node_id": "smoke-check",
                "role": "client",
                "event_type": "node_heartbeat",
                "status": "smoke",
                "details": {"source": "docker_smoke.py"},
            }
            asyncio.run(
                _assert_ws_broadcast(
                    monitor_url=args.monitor_url,
                    timeout_s=args.poll_interval_s * 5,
                    event_payload=ws_event_payload,
                )
            )
            _assert_control_sequence(
                client=client,
                monitor_url=args.monitor_url,
                run_id=control_run_id,
            )

        print("docker-smoke: ok")
        print(f"training-run-id: {args.training_run_id}")
        print(f"control-run-id: {control_run_id}")
        return 0
    finally:
        if cleanup:
            _run_compose(
                compose_file=args.compose_file,
                project_name=args.project_name,
                args=["down", "--remove-orphans"],
            )


if __name__ == "__main__":
    raise SystemExit(main())
