from __future__ import annotations

import json

import numpy as np
from flwr.common import ndarrays_to_parameters

from src.server.artifacts import ArtifactWriter
from src.server.config import ServerConfig


def test_artifact_writer_persists_round_metrics_and_checkpoint(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    params = ndarrays_to_parameters([np.array([1.0, 2.0, 3.0])])

    writer.write_round_metrics(
        round_number=1,
        num_clients=3,
        num_examples=90,
        aggregated_metrics={"loss": 0.42, "accuracy": 0.88},
    )
    checkpoint = writer.write_checkpoint(1, params)

    assert checkpoint is not None
    assert checkpoint.exists()
    lines = (tmp_path / "round_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["round"] == 1
    assert payload["aggregated_metrics"]["accuracy"] == 0.88


def test_artifact_writer_writes_server_config(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    cfg = ServerConfig(output_dir=tmp_path, rounds=7, run_id="test-run")

    writer.write_config(cfg)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["rounds"] == 7
    assert payload["run_id"] == "test-run"
