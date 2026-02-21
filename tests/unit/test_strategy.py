from __future__ import annotations

from pathlib import Path

import pytest

from src.server.artifacts import ArtifactWriter
from src.server.config import ServerConfig
from src.server.strategy import InstrumentedFedAvg, default_fit_metrics_aggregation_fn


class _FakeMonitor:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit_event(self, **kwargs) -> None:
        self.events.append(kwargs)


class _FakeClientManager:
    def __init__(self, available: int) -> None:
        self.available = available

    def num_available(self) -> int:
        return self.available


class _FakeFitRes:
    def __init__(self, num_examples: int) -> None:
        self.num_examples = num_examples


def test_default_fit_metrics_aggregation_fn_weighted_average() -> None:
    metrics = [
        (5, {"loss": 2.0, "accuracy": 0.4}),
        (15, {"loss": 1.0, "accuracy": 0.8}),
    ]

    aggregated = default_fit_metrics_aggregation_fn(metrics)

    assert aggregated["loss"] == 1.25
    assert aggregated["accuracy"] == 0.7


def test_default_fit_metrics_aggregation_fn_ignores_non_numeric_values() -> None:
    metrics = [
        (2, {"loss": 1.0, "note": "client-a"}),
        (2, {"loss": 3.0, "note": "client-b"}),
    ]

    aggregated = default_fit_metrics_aggregation_fn(metrics)

    assert aggregated == {"loss": 2.0}


def test_instrumented_strategy_emits_round_and_aggregation_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monitor = _FakeMonitor()
    strategy = InstrumentedFedAvg(
        config=ServerConfig(output_dir=tmp_path / "server"),
        artifact_writer=ArtifactWriter(tmp_path / "server"),
        monitor=monitor,
    )

    monkeypatch.setattr(
        "flwr.server.strategy.fedavg.FedAvg.configure_fit",
        lambda self, server_round, parameters, client_manager: [object(), object()],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "flwr.server.strategy.fedavg.FedAvg.aggregate_fit",
        lambda self, server_round, results, failures: (None, {"loss": 0.5}),  # noqa: ARG005
    )

    strategy.configure_fit(1, None, _FakeClientManager(available=3))
    strategy.aggregate_fit(1, [(None, _FakeFitRes(5)), (None, _FakeFitRes(7))], [])

    event_types = [str(event["event_type"]) for event in monitor.events]
    assert event_types == [
        "round_started",
        "model_dispatched",
        "aggregation_started",
        "aggregation_completed",
        "round_completed",
    ]


def test_instrumented_strategy_emits_node_error_when_configure_fit_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monitor = _FakeMonitor()
    strategy = InstrumentedFedAvg(
        config=ServerConfig(output_dir=tmp_path / "server"),
        artifact_writer=ArtifactWriter(tmp_path / "server"),
        monitor=monitor,
    )

    def _raise(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("configure failed")

    monkeypatch.setattr("flwr.server.strategy.fedavg.FedAvg.configure_fit", _raise)

    with pytest.raises(RuntimeError, match="configure failed"):
        strategy.configure_fit(1, None, _FakeClientManager(available=3))

    assert any(event["event_type"] == "node_error" for event in monitor.events)
