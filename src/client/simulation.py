"""Phase-aware local federated simulation runner for HF Flower clients."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

from src.client.hf_client import HFClientConfig, HFVisionClient
from src.ml.config import load_manifest, resolve_run_config
from src.server.artifacts import ArtifactWriter
from src.server.config import ServerConfig
from src.server.strategy import default_fit_metrics_aggregation_fn


@dataclass(frozen=True)
class RoundSummary:
    """Metrics summary for one federated round."""

    round: int
    aggregate_eval_loss: float
    client_train_examples: dict[str, int]
    client_eval_examples: dict[str, int]
    client_train_metrics: dict[str, dict[str, float]]
    client_eval_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SimulationSummary:
    """End-to-end simulation summary."""

    dataset_id: str
    phase: str
    rounds: int
    num_clients: int
    train_mode: str
    device: str
    model_id: str
    image_size: int
    round_summaries: list[RoundSummary]


class _SyntheticVisionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        num_examples: int,
        num_labels: int,
        image_size: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.pixel_values = torch.rand(
            num_examples,
            3,
            image_size,
            image_size,
            generator=generator,
        )
        self.labels = torch.tensor(
            [idx % num_labels for idx in range(num_examples)],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": self.pixel_values[index],
            "labels": self.labels[index],
        }


def _set_global_seeds(seed: int) -> None:
    """Set deterministic global RNG seeds for repeatable simulation runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _aggregate_fedavg(
    fit_results: list[tuple[str, list[np.ndarray], int]],
) -> list[np.ndarray]:
    total_examples = sum(num_examples for _, _, num_examples in fit_results)
    if total_examples <= 0:
        raise ValueError("Cannot aggregate with zero examples")

    aggregated: list[np.ndarray] = []
    per_client_weights = [weights for _, weights, _ in fit_results]
    per_client_examples = [num_examples for _, _, num_examples in fit_results]

    for layer_weights in zip(*per_client_weights, strict=True):
        first = np.asarray(layer_weights[0])
        if np.issubdtype(first.dtype, np.floating):
            weighted = np.zeros_like(first, dtype=np.float64)
            for client_layer, num_examples in zip(layer_weights, per_client_examples, strict=True):
                weighted += np.asarray(client_layer, dtype=np.float64) * num_examples
            aggregated.append((weighted / total_examples).astype(first.dtype))
        else:
            aggregated.append(first.copy())

    return aggregated


def _aggregate_numeric_metrics(
    entries: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    aggregated = default_fit_metrics_aggregation_fn(entries)
    return {
        key: float(value)
        for key, value in aggregated.items()
        if isinstance(value, (int, float))
    }


def _build_clients(
    *,
    dataset_id: str,
    num_labels: int,
    num_clients: int,
    output_dir: Path,
    model_id: str,
    requested_device: str,
    train_mode: str,
    image_size: int,
    train_examples_per_client: int,
    eval_examples: int,
    seed: int,
) -> list[HFVisionClient]:
    clients: list[HFVisionClient] = []
    label_names = [f"label_{idx}" for idx in range(num_labels)]

    for client_idx in range(num_clients):
        client_id = f"client_{client_idx}"
        train_dataset = _SyntheticVisionDataset(
            num_examples=train_examples_per_client,
            num_labels=num_labels,
            image_size=image_size,
            seed=seed + client_idx,
        )
        eval_dataset = _SyntheticVisionDataset(
            num_examples=eval_examples,
            num_labels=num_labels,
            image_size=image_size,
            seed=seed + 1000 + client_idx,
        )
        clients.append(
            HFVisionClient(
                config=HFClientConfig(
                    client_id=client_id,
                    dataset_id=dataset_id,
                    num_labels=num_labels,
                    output_dir=output_dir,
                    model_id=model_id,
                    requested_device=requested_device,
                    default_train_mode=train_mode,
                ),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                label_names=label_names,
            )
        )

    return clients


def run_local_simulation(
    *,
    dataset_id: str,
    output_dir: Path,
    model_id: str = "facebook/deit-tiny-patch16-224",
    requested_device: str = "cpu",
    rounds: int | None = None,
    num_clients: int | None = None,
    train_mode: str | None = None,
    image_size: int = 224,
    train_examples_per_client: int = 16,
    eval_examples: int = 8,
) -> SimulationSummary:
    """Run deterministic in-process federated simulation and return summary."""

    manifest = load_manifest(dataset_id)
    resolved = resolve_run_config(
        dataset_id,
        requested_device=requested_device,  # type: ignore[arg-type]
        train_mode=train_mode,  # type: ignore[arg-type]
        mps_available=False,
        enable_mps=requested_device == "mps",
    )

    effective_rounds = rounds if rounds is not None else resolved.rounds
    effective_clients = num_clients if num_clients is not None else resolved.num_clients
    effective_train_mode = train_mode if train_mode is not None else resolved.train_mode

    _set_global_seeds(resolved.run_seed)

    clients = _build_clients(
        dataset_id=dataset_id,
        num_labels=manifest.num_labels,
        num_clients=effective_clients,
        output_dir=output_dir,
        model_id=model_id,
        requested_device=resolved.device,
        train_mode=effective_train_mode,
        image_size=image_size,
        train_examples_per_client=train_examples_per_client,
        eval_examples=eval_examples,
        seed=resolved.run_seed,
    )

    global_parameters = clients[0].get_parameters({})
    round_summaries: list[RoundSummary] = []
    server_output_dir = output_dir / "server"
    artifact_writer = ArtifactWriter(server_output_dir)
    artifact_writer.write_config(
        ServerConfig(
            rounds=effective_rounds,
            min_fit_clients=effective_clients,
            min_evaluate_clients=effective_clients,
            min_available_clients=effective_clients,
            output_dir=server_output_dir,
        )
    )

    for round_idx in range(1, effective_rounds + 1):
        fit_results: list[tuple[str, list[np.ndarray], int]] = []
        client_train_metrics: dict[str, dict[str, float]] = {}
        client_train_examples: dict[str, int] = {}

        for client in clients:
            updated, train_count, train_metrics = client.fit(
                global_parameters,
                {
                    "round": round_idx,
                    "train_mode": effective_train_mode,
                },
            )
            client_id = client.config.client_id
            fit_results.append((client_id, updated, train_count))
            client_train_examples[client_id] = train_count
            client_train_metrics[client_id] = train_metrics

        global_parameters = _aggregate_fedavg(fit_results)

        client_eval_metrics: dict[str, dict[str, float]] = {}
        client_eval_examples: dict[str, int] = {}
        eval_losses: list[float] = []
        for client in clients:
            loss, eval_count, eval_metrics = client.evaluate(
                global_parameters,
                {
                    "round": round_idx,
                    "train_mode": effective_train_mode,
                },
            )
            client_id = client.config.client_id
            client_eval_examples[client_id] = eval_count
            client_eval_metrics[client_id] = eval_metrics
            eval_losses.append(loss)

        round_summary = RoundSummary(
            round=round_idx,
            aggregate_eval_loss=float(np.mean(eval_losses)),
            client_train_examples=client_train_examples,
            client_eval_examples=client_eval_examples,
            client_train_metrics=client_train_metrics,
            client_eval_metrics=client_eval_metrics,
        )
        round_summaries.append(round_summary)

        aggregated_train_metrics = _aggregate_numeric_metrics(
            [
                (client_train_examples[client_id], client_train_metrics[client_id])
                for client_id in sorted(client_train_examples)
            ]
        )
        aggregated_eval_metrics = _aggregate_numeric_metrics(
            [
                (client_eval_examples[client_id], client_eval_metrics[client_id])
                for client_id in sorted(client_eval_examples)
            ]
        )
        artifact_writer.write_round_metrics(
            round_number=round_idx,
            num_clients=len(clients),
            num_examples=sum(client_train_examples.values()),
            aggregated_metrics={
                "aggregate_eval_loss": round_summary.aggregate_eval_loss,
                "train": aggregated_train_metrics,
                "eval": aggregated_eval_metrics,
            },
        )
        artifact_writer.write_checkpoint(
            round_number=round_idx,
            parameters=ndarrays_to_parameters(global_parameters),
        )

    return SimulationSummary(
        dataset_id=dataset_id,
        phase=manifest.phase,
        rounds=effective_rounds,
        num_clients=effective_clients,
        train_mode=effective_train_mode,
        device=resolved.device,
        model_id=model_id,
        image_size=image_size,
        round_summaries=round_summaries,
    )


def write_simulation_summary(summary: SimulationSummary, *, output_path: Path) -> Path:
    """Persist simulation summary artifacts."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
