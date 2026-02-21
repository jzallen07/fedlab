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
    torch.manual_seed(777)
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


def test_run_local_simulation_and_write_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.client.hf_client.build_deit_model", _build_fake_model)
    monkeypatch.setattr("src.ml.trainer.build_deit_model", _build_fake_model)
    monkeypatch.setattr("src.ml.trainer.build_compute_metrics_fn", _compute_metrics_callback)

    summary = run_local_simulation(
        dataset_id="bloodmnist",
        output_dir=tmp_path / "simulation",
        model_id="fake/model",
        requested_device="cpu",
        rounds=1,
        num_clients=2,
        train_mode="head_only",
        image_size=16,
        train_examples_per_client=4,
        eval_examples=2,
    )

    assert summary.dataset_id == "bloodmnist"
    assert summary.rounds == 1
    assert summary.num_clients == 2
    assert len(summary.round_summaries) == 1
    assert summary.round_summaries[0].aggregate_eval_loss >= 0.0

    output = write_simulation_summary(summary, output_path=tmp_path / "summary" / "sim.json")
    assert output.exists()
