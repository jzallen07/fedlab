"""Flower strategy wiring with server-side instrumentation hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flwr.common import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.monitor.emitter import MonitorEmitter
from src.server.artifacts import ArtifactWriter
from src.server.config import ServerConfig

MetricsAggFn = Callable[[list[tuple[int, dict[str, Scalar]]]], dict[str, Scalar]]


def default_fit_metrics_aggregation_fn(
    metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """Aggregate client metrics by weighting numeric fields with sample counts."""
    if not metrics:
        return {}

    totals: dict[str, float] = {}
    total_examples = 0

    for num_examples, entry in metrics:
        total_examples += num_examples
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value) * num_examples

    if total_examples == 0:
        return {}
    return {key: value / total_examples for key, value in totals.items()}


class InstrumentedFedAvg(FedAvg):
    """FedAvg strategy with telemetry and artifact writing hooks."""

    def __init__(
        self,
        *,
        config: ServerConfig,
        artifact_writer: ArtifactWriter,
        monitor: MonitorEmitter | None = None,
        fit_metrics_aggregation_fn: MetricsAggFn = default_fit_metrics_aggregation_fn,
    ) -> None:
        super().__init__(
            fraction_fit=config.fraction_fit,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=config.min_fit_clients,
            min_evaluate_clients=config.min_evaluate_clients,
            min_available_clients=config.min_available_clients,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
        self.config = config
        self.artifact_writer = artifact_writer
        self.monitor = monitor

    def configure_fit(self, server_round, parameters, client_manager):
        try:
            self._emit(
                event_type="round_started",
                round_number=server_round,
                status="running",
                metrics={"configured_clients": client_manager.num_available()},
            )
            fit_cfg = super().configure_fit(server_round, parameters, client_manager)
            self._emit(
                event_type="model_dispatched",
                round_number=server_round,
                status="running",
                metrics={"selected_clients": len(fit_cfg)},
            )
            return fit_cfg
        except Exception as exc:
            self._emit(
                event_type="node_error",
                round_number=server_round,
                status="error",
                details={"stage": "server.configure_fit", "error": str(exc)},
            )
            raise

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ):
        try:
            self._emit(
                event_type="aggregation_started",
                round_number=server_round,
                status="aggregating",
                metrics={"result_count": len(results), "failure_count": len(failures)},
            )
            params, metrics = super().aggregate_fit(server_round, results, failures)
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            aggregated_metrics: dict[str, Any] = dict(metrics or {})
            self.artifact_writer.write_round_metrics(
                round_number=server_round,
                num_clients=len(results),
                num_examples=total_examples,
                aggregated_metrics=aggregated_metrics,
            )
            checkpoint_path = self.artifact_writer.write_checkpoint(server_round, params)
            self._emit(
                event_type="aggregation_completed",
                round_number=server_round,
                status="running",
                metrics={
                    "result_count": len(results),
                    "failure_count": len(failures),
                    "num_examples": total_examples,
                    "checkpoint_written": 1 if checkpoint_path else 0,
                },
            )
            self._emit(
                event_type="round_completed",
                round_number=server_round,
                status="running",
                metrics=aggregated_metrics,
            )
            return params, metrics
        except Exception as exc:
            self._emit(
                event_type="node_error",
                round_number=server_round,
                status="error",
                details={"stage": "server.aggregate_fit", "error": str(exc)},
            )
            raise

    def _emit(
        self,
        *,
        event_type: str,
        round_number: int,
        status: str,
        latency_ms: float | None = None,
        payload_bytes: int | None = None,
        metrics: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.monitor is None:
            return
        self.monitor.emit_event(
            event_type=event_type,
            run_id=self.config.run_id,
            node_id="server",
            role="server",
            round_number=round_number,
            status=status,
            latency_ms=latency_ms,
            payload_bytes=payload_bytes,
            metrics=metrics,
            details=details,
        )


def build_strategy(
    *,
    config: ServerConfig,
    artifact_writer: ArtifactWriter,
    monitor: MonitorEmitter | None = None,
) -> InstrumentedFedAvg:
    """Build an instrumented FedAvg strategy from server configuration."""
    return InstrumentedFedAvg(
        config=config,
        artifact_writer=artifact_writer,
        monitor=monitor,
    )
