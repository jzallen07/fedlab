"""Model state serialization helpers for Flower parameter exchange."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel


def model_to_ndarrays(model: PreTrainedModel) -> list[np.ndarray]:
    """Serialize model state dict tensors to numpy arrays."""

    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def ndarrays_to_model(model: PreTrainedModel, weights: Iterable[np.ndarray]) -> None:
    """Load numpy-array weights into a model state dict."""

    state_dict = model.state_dict()
    weight_list = list(weights)
    if len(weight_list) != len(state_dict):
        raise ValueError(
            f"Weight count mismatch: expected {len(state_dict)}, received {len(weight_list)}"
        )

    restored = OrderedDict()
    for (name, tensor), array in zip(state_dict.items(), weight_list, strict=True):
        candidate = torch.from_numpy(np.asarray(array))
        if tuple(candidate.shape) != tuple(tensor.shape):
            raise ValueError(
                "Shape mismatch for "
                f"'{name}': expected {tuple(tensor.shape)}, got {tuple(candidate.shape)}"
            )
        restored[name] = candidate.to(dtype=tensor.dtype)

    model.load_state_dict(restored, strict=True)
