from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import DeiTConfig, DeiTForImageClassification

from src.client.hf_client import HFClientConfig, HFVisionClient
from src.ml.serialization import model_to_ndarrays, ndarrays_to_model


class _TinyVisionDataset(torch.utils.data.Dataset):
    def __init__(self, *, num_examples: int, num_labels: int, image_size: int = 16) -> None:
        generator = torch.Generator().manual_seed(456)
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


class _FakeMonitor:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit_event(self, **kwargs) -> None:
        self.events.append(kwargs)


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
    return _tiny_deit_model(num_labels=num_labels)


def _compute_metrics_callback():
    def _callback(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted = np.asarray(logits).argmax(axis=-1)
        references = np.asarray(eval_prediction.label_ids)
        return {"accuracy": float((predicted == references).mean())}

    return _callback


def test_model_serialization_roundtrip() -> None:
    model = _tiny_deit_model(num_labels=3)
    weights = model_to_ndarrays(model)

    restored = _tiny_deit_model(num_labels=3)
    ndarrays_to_model(restored, weights)

    for tensor_a, tensor_b in zip(
        model.state_dict().values(),
        restored.state_dict().values(),
        strict=True,
    ):
        np.testing.assert_allclose(tensor_a.detach().cpu().numpy(), tensor_b.detach().cpu().numpy())


def test_hf_client_fit_and_evaluate_emit_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.client.hf_client.build_deit_model", _build_fake_model)
    monkeypatch.setattr("src.ml.trainer.build_deit_model", _build_fake_model)
    monkeypatch.setattr("src.ml.trainer.build_compute_metrics_fn", _compute_metrics_callback)

    train_dataset = _TinyVisionDataset(num_examples=8, num_labels=3)
    eval_dataset = _TinyVisionDataset(num_examples=4, num_labels=3)
    monitor = _FakeMonitor()

    client = HFVisionClient(
        config=HFClientConfig(
            client_id="client_0",
            dataset_id="bloodmnist",
            num_labels=3,
            output_dir=tmp_path / "client",
            model_id="fake/model",
            requested_device="cpu",
            default_train_mode="head_only",
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        label_names=["a", "b", "c"],
        monitor=monitor,
    )

    initial = client.get_parameters({})
    assert len(initial) > 0

    updated, num_train_examples, train_metrics = client.fit(
        initial,
        {
            "round": 1,
            "train_mode": "head_only",
        },
    )
    assert num_train_examples == len(train_dataset)
    assert len(updated) == len(initial)
    assert "train_runtime" in train_metrics
    assert "eval_loss" in train_metrics

    loss, num_eval_examples, eval_metrics = client.evaluate(
        updated,
        {
            "train_mode": "head_only",
        },
    )
    assert isinstance(loss, float)
    assert num_eval_examples == len(eval_dataset)
    assert "eval_loss" in eval_metrics
    event_types = [str(event["event_type"]) for event in monitor.events]
    assert "node_heartbeat" in event_types
    assert "client_train_started" in event_types
    assert "client_train_completed" in event_types
    assert "client_update_uploaded" in event_types


def test_hf_client_fit_emits_node_error_on_trainer_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.client.hf_client.build_deit_model", _build_fake_model)

    def _raise(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("trainer failed")

    monkeypatch.setattr("src.client.hf_client.create_trainer", _raise)
    monitor = _FakeMonitor()

    client = HFVisionClient(
        config=HFClientConfig(
            client_id="client_0",
            dataset_id="bloodmnist",
            num_labels=3,
            output_dir=tmp_path / "client",
            model_id="fake/model",
            requested_device="cpu",
            default_train_mode="head_only",
        ),
        train_dataset=_TinyVisionDataset(num_examples=8, num_labels=3),
        eval_dataset=_TinyVisionDataset(num_examples=4, num_labels=3),
        monitor=monitor,
    )

    with pytest.raises(RuntimeError, match="trainer failed"):
        client.fit([], {"round": 1, "train_mode": "head_only"})

    assert any(event["event_type"] == "node_error" for event in monitor.events)
