from pathlib import Path

import pytest
import torch
from transformers import DeiTConfig, DeiTForImageClassification

from src.ml.model import count_trainable_parameters
from src.ml.trainer import create_trainer, run_trainer_round


class _TinyVisionDataset(torch.utils.data.Dataset):
    def __init__(self, *, num_examples: int, num_labels: int, image_size: int = 16) -> None:
        generator = torch.Generator().manual_seed(123)
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


def _tiny_deit_model(num_labels: int) -> DeiTForImageClassification:
    config = DeiTConfig(
        num_labels=num_labels,
        image_size=16,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    return DeiTForImageClassification(config)


def _build_fake_model(
    num_labels: int,
    *,
    model_id: str,
    label_names: list[str] | None,
) -> DeiTForImageClassification:
    del model_id, label_names
    return _tiny_deit_model(num_labels)


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


def test_create_trainer_head_only_cpu_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.ml.trainer.build_deit_model", _build_fake_model)
    monkeypatch.setattr("src.ml.trainer.build_compute_metrics_fn", _compute_metrics_callback)

    train_dataset = _TinyVisionDataset(num_examples=8, num_labels=3)
    eval_dataset = _TinyVisionDataset(num_examples=4, num_labels=3)

    trainer, resolved = create_trainer(
        dataset_id="bloodmnist",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_labels=3,
        label_names=["a", "b", "c"],
        output_dir=tmp_path / "trainer",
        requested_device="cpu",
        train_mode="head_only",
        mps_available=False,
        enable_mps=False,
        model_id="fake/model",
    )

    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameters = count_trainable_parameters(trainer.model)

    assert resolved.device == "cpu"
    assert resolved.train_mode == "head_only"
    assert trainable_parameters > 0
    assert trainable_parameters < total_parameters


def test_run_trainer_round_smoke_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.ml.trainer.build_deit_model", _build_fake_model)
    monkeypatch.setattr("src.ml.trainer.build_compute_metrics_fn", _compute_metrics_callback)

    train_dataset = _TinyVisionDataset(num_examples=8, num_labels=3)
    eval_dataset = _TinyVisionDataset(num_examples=4, num_labels=3)

    result = run_trainer_round(
        dataset_id="bloodmnist",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_labels=3,
        label_names=["a", "b", "c"],
        output_dir=tmp_path / "round",
        requested_device="cpu",
        train_mode="head_only",
        mps_available=False,
        enable_mps=False,
        model_id="fake/model",
    )

    assert result.resolved_config.device == "cpu"
    assert result.resolved_config.train_mode == "head_only"
    assert result.trainable_parameters > 0
    assert "train_runtime" in result.train_metrics
    assert "eval_loss" in result.eval_metrics
