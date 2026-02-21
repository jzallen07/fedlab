"""Flower client wrapper over the HF trainer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import flwr as fl
import numpy as np
from transformers import PreTrainedModel

from src.ml.model import build_deit_model
from src.ml.serialization import model_to_ndarrays, ndarrays_to_model
from src.ml.trainer import create_trainer
from src.monitor.emitter import MonitorEmitter


@dataclass(frozen=True)
class HFClientConfig:
    """Runtime configuration for a single HF Flower client."""

    client_id: str
    dataset_id: str
    num_labels: int
    output_dir: Path
    model_id: str = "facebook/deit-tiny-patch16-224"
    requested_device: str = "auto"
    default_train_mode: str = "head_only"
    run_id: str = "local-run"
    monitor_url: str | None = None
    monitor_timeout_s: float = 2.0


class HFVisionClient(fl.client.NumPyClient):
    """NumPyClient implementation backed by transformers.Trainer."""

    def __init__(
        self,
        *,
        config: HFClientConfig,
        train_dataset: Any,
        eval_dataset: Any,
        label_names: list[str] | None = None,
        monitor: MonitorEmitter | None = None,
    ) -> None:
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.label_names = label_names
        self.monitor = monitor or MonitorEmitter(
            base_url=config.monitor_url,
            timeout_s=config.monitor_timeout_s,
        )

        self._initial_model: PreTrainedModel = build_deit_model(
            num_labels=config.num_labels,
            model_id=config.model_id,
            label_names=label_names,
        )
        self._emit(
            event_type="node_heartbeat",
            status="ready",
            metrics={
                "train_examples": len(self.train_dataset),
                "eval_examples": len(self.eval_dataset),
            },
            details={"dataset_id": self.config.dataset_id},
        )

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        del config
        return model_to_ndarrays(self._initial_model)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        train_mode = str(config.get("train_mode", self.config.default_train_mode))
        round_idx = int(config.get("round", 0))
        self._emit(
            event_type="client_train_started",
            round_number=round_idx,
            status="running",
            metrics={"train_examples": len(self.train_dataset)},
            details={"train_mode": train_mode},
        )
        started = perf_counter()

        try:
            round_output_dir = self.config.output_dir / self.config.client_id / f"round_{round_idx}"
            trainer, _ = create_trainer(
                dataset_id=self.config.dataset_id,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                num_labels=self.config.num_labels,
                label_names=self.label_names,
                output_dir=round_output_dir,
                requested_device=self.config.requested_device,  # type: ignore[arg-type]
                train_mode=train_mode,  # type: ignore[arg-type]
                model_id=self.config.model_id,
            )

            ndarrays_to_model(trainer.model, parameters)
            train_output = trainer.train()
            eval_metrics = trainer.evaluate(eval_dataset=self.eval_dataset)
        except Exception as exc:
            self._emit(
                event_type="node_error",
                round_number=round_idx,
                status="error",
                details={
                    "stage": "client.fit",
                    "error": str(exc),
                },
            )
            raise

        metrics: dict[str, float] = {}
        for key, value in train_output.metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        updated_parameters = model_to_ndarrays(trainer.model)
        num_examples = len(self.train_dataset)
        latency_ms = (perf_counter() - started) * 1000.0
        payload_bytes = int(sum(weights.nbytes for weights in updated_parameters))
        self._emit(
            event_type="client_train_completed",
            round_number=round_idx,
            status="completed",
            latency_ms=latency_ms,
            metrics=metrics,
            details={"train_mode": train_mode},
        )
        self._emit(
            event_type="client_update_uploaded",
            round_number=round_idx,
            status="uploaded",
            payload_bytes=payload_bytes,
            metrics={"train_examples": num_examples},
        )
        return updated_parameters, int(num_examples), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        train_mode = str(config.get("train_mode", self.config.default_train_mode))

        output_dir = self.config.output_dir / self.config.client_id / "eval"
        try:
            trainer, _ = create_trainer(
                dataset_id=self.config.dataset_id,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                num_labels=self.config.num_labels,
                label_names=self.label_names,
                output_dir=output_dir,
                requested_device=self.config.requested_device,  # type: ignore[arg-type]
                train_mode=train_mode,  # type: ignore[arg-type]
                model_id=self.config.model_id,
            )

            ndarrays_to_model(trainer.model, parameters)
            eval_metrics = trainer.evaluate(eval_dataset=self.eval_dataset)
        except Exception as exc:
            self._emit(
                event_type="node_error",
                status="error",
                details={
                    "stage": "client.evaluate",
                    "error": str(exc),
                },
            )
            raise

        metrics: dict[str, float] = {}
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        loss = float(metrics.get("eval_loss", 0.0))
        num_examples = len(self.eval_dataset)
        return loss, int(num_examples), metrics

    def _emit(
        self,
        *,
        event_type: str,
        status: str,
        round_number: int | None = None,
        latency_ms: float | None = None,
        payload_bytes: int | None = None,
        metrics: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.monitor.emit_event(
            event_type=event_type,
            run_id=self.config.run_id,
            node_id=self.config.client_id,
            role="client",
            status=status,
            round_number=round_number,
            latency_ms=latency_ms,
            payload_bytes=payload_bytes,
            metrics=metrics,
            details=details,
        )


def build_numpy_client(
    *,
    config: HFClientConfig,
    train_dataset: Any,
    eval_dataset: Any,
    label_names: list[str] | None = None,
    monitor: MonitorEmitter | None = None,
) -> fl.client.NumPyClient:
    """Factory for Flower client integration."""

    return HFVisionClient(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        label_names=label_names,
        monitor=monitor,
    )
