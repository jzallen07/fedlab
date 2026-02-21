"""DeiT model construction and train-mode controls for federated clients."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch.nn as nn
from transformers import AutoModelForImageClassification, PreTrainedModel

MODEL_ID_DEIT_TINY = "facebook/deit-tiny-patch16-224"
TrainMode = Literal["head_only", "unfreeze_last_block"]


def _load_pretrained_model(
    model_id: str,
    *,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> PreTrainedModel:
    """Load a DeiT image-classification model with classifier replacement."""

    return AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def _get_classifier_module(model: PreTrainedModel) -> nn.Module:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Module):
        raise ValueError("Model is missing classifier module")
    return classifier


def _get_encoder_layers(model: PreTrainedModel) -> Sequence[nn.Module]:
    deit = getattr(model, "deit", None)
    if deit is not None:
        encoder = getattr(deit, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layer", None)
            if layers is not None:
                return list(layers)

    vit = getattr(model, "vit", None)
    if vit is not None:
        encoder = getattr(vit, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layer", None)
            if layers is not None:
                return list(layers)

    raise ValueError("Model is missing recognizable transformer encoder layers")


def build_deit_model(
    num_labels: int,
    *,
    model_id: str = MODEL_ID_DEIT_TINY,
    label_names: Sequence[str] | None = None,
) -> PreTrainedModel:
    """Build a DeiT classifier with replaced head for a target label count."""

    if num_labels <= 1:
        raise ValueError("num_labels must be > 1")

    if label_names is not None and len(label_names) != num_labels:
        raise ValueError("label_names length must match num_labels")

    if label_names is None:
        label_names = [f"label_{idx}" for idx in range(num_labels)]

    id2label = {idx: name for idx, name in enumerate(label_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    model = _load_pretrained_model(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    classifier = _get_classifier_module(model)
    out_features = getattr(classifier, "out_features", None)
    if out_features != num_labels:
        raise ValueError(
            f"Classifier output size mismatch: expected {num_labels}, got {out_features}"
        )

    return model


def apply_train_mode(model: PreTrainedModel, mode: TrainMode) -> None:
    """Apply trainability policy for FL rounds."""

    if mode not in {"head_only", "unfreeze_last_block"}:
        raise ValueError(f"Unsupported train mode: {mode}")

    _set_requires_grad(model, False)

    classifier = _get_classifier_module(model)
    _set_requires_grad(classifier, True)

    if mode == "unfreeze_last_block":
        layers = _get_encoder_layers(model)
        if not layers:
            raise ValueError("No encoder layers found for unfreeze_last_block mode")
        _set_requires_grad(layers[-1], True)


def count_trainable_parameters(model: PreTrainedModel) -> int:
    """Return number of trainable parameters in model."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
