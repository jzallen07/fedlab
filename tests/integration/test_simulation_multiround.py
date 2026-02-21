from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import DeiTConfig, DeiTForImageClassification

from src.client.simulation import run_local_simulation, write_simulation_summary


def _tiny_deit_model(num_labels: int) -> DeiTForImageClassification:
    return DeiTForImageClassification(
        DeiTConfig(
            num_labels=num_labels,
            image_size=16,
            patch_size=16,
            num_channels=3,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
    )


def _build_fake_model(
    num_labels: int,
    *,
    model_id: str,
    label_names: list[str] | None,
) -> DeiTForImageClassification:
    del model_id, label_names
    torch.manual_seed(20260221)
    return _tiny_deit_model(num_labels=num_labels)


def _compute_metrics_callback():
    def _callback(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted = torch.tensor(logits).argmax(dim=-1).numpy()
        references = eval_prediction.label_ids
        accuracy = float((predicted == references).mean())
        return {"accuracy": accuracy}

    return _callback


def test_simulation_multiround_persists_artifacts_per_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.client.hf_client.build_deit_model", _build_fake_model)
    monkeypatch.setattr("src.ml.trainer.build_deit_model", _build_fake_model)
    monkeypatch.setattr("src.ml.trainer.build_compute_metrics_fn", _compute_metrics_callback)

    simulation_dir = tmp_path / "simulation-multiround"
    summary = run_local_simulation(
        dataset_id="bloodmnist",
        output_dir=simulation_dir,
        model_id="fake/model",
        requested_device="cpu",
        rounds=2,
        num_clients=2,
        train_mode="head_only",
        image_size=16,
        train_examples_per_client=4,
        eval_examples=2,
    )
    summary_path = write_simulation_summary(summary, output_path=simulation_dir / "summary.json")

    assert summary_path.exists()
    assert summary.rounds == 2
    assert len(summary.round_summaries) == 2

    server_dir = simulation_dir / "server"
    server_config = json.loads((server_dir / "summary.json").read_text(encoding="utf-8"))
    assert server_config["rounds"] == 2
    assert server_config["min_fit_clients"] == 2
    assert server_config["min_evaluate_clients"] == 2
    assert server_config["min_available_clients"] == 2

    lines = (server_dir / "round_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert [payload["round"] for payload in payloads] == [1, 2]
    assert all(payload["num_clients"] == 2 for payload in payloads)
    assert all(payload["num_examples"] == 8 for payload in payloads)
    assert all("train" in payload["aggregated_metrics"] for payload in payloads)
    assert all("eval" in payload["aggregated_metrics"] for payload in payloads)
    assert all(
        payload["aggregated_metrics"]["aggregate_eval_loss"] >= 0.0 for payload in payloads
    )

    assert (server_dir / "round-0001.npz").exists()
    assert (server_dir / "round-0002.npz").exists()
