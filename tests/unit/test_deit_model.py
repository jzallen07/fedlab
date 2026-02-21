import pytest
import torch
from transformers import DeiTConfig, DeiTForImageClassification

from src.ml.model import apply_train_mode, build_deit_model, count_trainable_parameters


def _build_tiny_deit(num_labels: int) -> DeiTForImageClassification:
    config = DeiTConfig(
        num_labels=num_labels,
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    return DeiTForImageClassification(config)


def test_build_deit_model_sets_classifier_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader(
        model_id: str,
        *,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
    ) -> DeiTForImageClassification:
        del model_id
        model = _build_tiny_deit(num_labels=num_labels)
        model.config.id2label = id2label
        model.config.label2id = label2id
        return model

    monkeypatch.setattr("src.ml.model._load_pretrained_model", fake_loader)

    model = build_deit_model(num_labels=5)

    assert model.classifier.out_features == 5
    assert model.config.num_labels == 5
    assert model.config.id2label[0] == "label_0"


def test_forward_shape_matches_batch_and_label_count(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader(
        model_id: str,
        *,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
    ) -> DeiTForImageClassification:
        del model_id, id2label, label2id
        return _build_tiny_deit(num_labels=num_labels)

    monkeypatch.setattr("src.ml.model._load_pretrained_model", fake_loader)

    model = build_deit_model(num_labels=4)
    outputs = model(pixel_values=torch.randn(2, 3, 32, 32))

    assert tuple(outputs.logits.shape) == (2, 4)


def test_apply_train_mode_head_only_and_unfreeze_last_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_loader(
        model_id: str,
        *,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
    ) -> DeiTForImageClassification:
        del model_id, id2label, label2id
        return _build_tiny_deit(num_labels=num_labels)

    monkeypatch.setattr("src.ml.model._load_pretrained_model", fake_loader)

    model = build_deit_model(num_labels=3)

    apply_train_mode(model, "head_only")
    head_only_trainable = count_trainable_parameters(model)

    assert all(param.requires_grad for param in model.classifier.parameters())
    assert all(
        not param.requires_grad
        for layer in model.deit.encoder.layer
        for param in layer.parameters()
    )

    apply_train_mode(model, "unfreeze_last_block")
    unfreeze_trainable = count_trainable_parameters(model)

    assert all(param.requires_grad for param in model.classifier.parameters())
    assert all(
        not param.requires_grad
        for param in model.deit.encoder.layer[0].parameters()
    )
    assert all(
        param.requires_grad
        for param in model.deit.encoder.layer[-1].parameters()
    )
    assert unfreeze_trainable > head_only_trainable


def test_invalid_label_names_length_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader(
        model_id: str,
        *,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
    ) -> DeiTForImageClassification:
        del model_id, id2label, label2id
        return _build_tiny_deit(num_labels=num_labels)

    monkeypatch.setattr("src.ml.model._load_pretrained_model", fake_loader)

    with pytest.raises(ValueError):
        build_deit_model(num_labels=4, label_names=["a", "b"])


def test_invalid_train_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader(
        model_id: str,
        *,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
    ) -> DeiTForImageClassification:
        del model_id, id2label, label2id
        return _build_tiny_deit(num_labels=num_labels)

    monkeypatch.setattr("src.ml.model._load_pretrained_model", fake_loader)

    model = build_deit_model(num_labels=2)

    with pytest.raises(ValueError):
        apply_train_mode(model, "bad-mode")  # type: ignore[arg-type]
