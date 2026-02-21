"""Server-side metrics and model artifact persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import Parameters, parameters_to_ndarrays

from src.server.config import ServerConfig


class ArtifactWriter:
    """Persist round-level metrics and model snapshots."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_metrics_path = self.output_dir / "round_metrics.jsonl"
        self.summary_path = self.output_dir / "summary.json"

    def write_config(self, config: ServerConfig) -> None:
        """Persist server config for experiment reproducibility."""
        payload = asdict(config)
        payload["output_dir"] = str(config.output_dir)
        self.summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_round_metrics(
        self,
        *,
        round_number: int,
        num_clients: int,
        num_examples: int,
        aggregated_metrics: dict[str, Any],
    ) -> None:
        """Append round metrics into JSONL output."""
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "round": round_number,
            "num_clients": num_clients,
            "num_examples": num_examples,
            "aggregated_metrics": aggregated_metrics,
        }
        with self.round_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_checkpoint(self, round_number: int, parameters: Parameters | None) -> Path | None:
        """Persist aggregated weights for the round as a NumPy archive."""
        if parameters is None:
            return None

        ndarrays = parameters_to_ndarrays(parameters)
        checkpoint_path = self.output_dir / f"round-{round_number:04d}.npz"
        np.savez(checkpoint_path, *ndarrays)
        return checkpoint_path
