"""Hugging Face preprocessing pipeline for MedMNIST datasets."""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import datasets.fingerprint as datasets_fingerprint
import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image
from transformers import AutoImageProcessor

from src.ml.data import DATASET_SPLITS, SUPPORTED_MEDMNIST_DATASETS
from src.ml.model import MODEL_ID_DEIT_TINY


def _pickle_dumps(value: Any) -> bytes:
    return pickle.dumps(value)


def _ensure_datasets_fingerprint_compat() -> None:
    """Work around datasets+dill incompatibility in Python 3.14."""

    if sys.version_info >= (3, 14):
        datasets_fingerprint.dumps = _pickle_dumps


@dataclass(frozen=True)
class PreprocessMetadata:
    """Metadata emitted for preprocessed MedMNIST datasets."""

    dataset_id: str
    model_id: str
    image_size: int
    split_sizes: dict[str, int]
    label2id: dict[str, int]
    id2label: dict[int, str]
    processor: dict[str, Any]


def _import_medmnist() -> ModuleType:
    try:
        import medmnist
    except ImportError as exc:
        raise RuntimeError(
            "medmnist is required for preprocessing. Install project dependencies first."
        ) from exc
    return medmnist


def _normalize_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.strip().lower()
    if normalized not in SUPPORTED_MEDMNIST_DATASETS:
        expected = ", ".join(SUPPORTED_MEDMNIST_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_id}'. Expected one of: {expected}")
    return normalized


def _resolve_dataset_info(medmnist: ModuleType, dataset_id: str) -> dict[str, Any]:
    info = medmnist.INFO.get(dataset_id)
    if not isinstance(info, dict):
        raise ValueError(f"Dataset metadata not found in medmnist.INFO: {dataset_id}")
    return info


def _resolve_dataset_class(medmnist: ModuleType, dataset_id: str) -> type[object]:
    info = _resolve_dataset_info(medmnist, dataset_id)
    class_name = info.get("python_class")
    if not isinstance(class_name, str):
        raise ValueError(f"Dataset metadata missing python_class for {dataset_id}")

    dataset_cls = getattr(medmnist, class_name, None)
    if dataset_cls is None:
        raise ValueError(f"Dataset class '{class_name}' not found in medmnist module")
    return dataset_cls


def resolve_label_names(dataset_id: str) -> list[str]:
    """Resolve label names from MedMNIST metadata."""

    normalized_id = _normalize_dataset_id(dataset_id)
    medmnist = _import_medmnist()
    info = _resolve_dataset_info(medmnist, normalized_id)

    raw_labels = info.get("label")
    if not isinstance(raw_labels, dict):
        raise ValueError(f"Dataset metadata missing label map for {normalized_id}")

    ordered_items = sorted(raw_labels.items(), key=lambda item: int(item[0]))
    return [str(label_name) for _, label_name in ordered_items]


def build_image_processor(model_id: str = MODEL_ID_DEIT_TINY):
    """Build HF image processor aligned with model normalization settings."""

    return AutoImageProcessor.from_pretrained(model_id)


def load_medmnist_split(
    dataset_id: str,
    *,
    split: str,
    root: Path,
    image_size: int,
    download: bool,
) -> tuple[list[np.ndarray], list[int]]:
    """Load one MedMNIST split and return images and integer labels."""

    if split not in DATASET_SPLITS:
        expected = ", ".join(DATASET_SPLITS)
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {expected}")

    normalized_id = _normalize_dataset_id(dataset_id)
    medmnist = _import_medmnist()
    dataset_cls = _resolve_dataset_class(medmnist, normalized_id)

    dataset = dataset_cls(split=split, root=str(root), size=image_size, download=download)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for index in range(len(dataset)):
        image, label = dataset[index]
        images.append(np.asarray(image))
        labels.append(int(np.asarray(label).reshape(-1)[0]))

    return images, labels


def build_hf_split(
    images: list[np.ndarray],
    labels: list[int],
    *,
    label_names: list[str],
    image_processor: Any,
) -> Dataset:
    """Convert image+label records into a processed HF dataset split."""

    _ensure_datasets_fingerprint_compat()

    class_label = ClassLabel(names=label_names)
    raw_dataset = Dataset.from_dict(
        {
            "image": images,
            "label": labels,
        },
        features=Features(
            {
                "image": Image(),
                "label": class_label,
            }
        ),
    )

    def _transform(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        encoded = image_processor(images=batch["image"], return_tensors="np")
        pixel_values = [np.asarray(array, dtype=np.float32) for array in encoded["pixel_values"]]
        return {
            "pixel_values": pixel_values,
            "labels": list(batch["label"]),
        }

    processed = raw_dataset.map(
        _transform,
        batched=True,
        remove_columns=["image", "label"],
    )

    processed.set_format(type="numpy", columns=["pixel_values", "labels"])
    return processed


def preprocess_medmnist_dataset(
    dataset_id: str,
    *,
    root: Path,
    model_id: str = MODEL_ID_DEIT_TINY,
    image_size: int = 224,
    download: bool = False,
    image_processor: Any | None = None,
) -> tuple[DatasetDict, PreprocessMetadata]:
    """Preprocess MedMNIST train/val/test splits into HF DatasetDict."""

    normalized_id = _normalize_dataset_id(dataset_id)
    label_names = resolve_label_names(normalized_id)
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for name, idx in label2id.items()}

    processor = image_processor if image_processor is not None else build_image_processor(model_id)

    splits: dict[str, Dataset] = {}
    split_sizes: dict[str, int] = {}
    for split in DATASET_SPLITS:
        images, labels = load_medmnist_split(
            normalized_id,
            split=split,
            root=root,
            image_size=image_size,
            download=download,
        )
        split_sizes[split] = len(labels)
        splits[split] = build_hf_split(
            images,
            labels,
            label_names=label_names,
            image_processor=processor,
        )

    metadata = PreprocessMetadata(
        dataset_id=normalized_id,
        model_id=model_id,
        image_size=image_size,
        split_sizes=split_sizes,
        label2id=label2id,
        id2label=id2label,
        processor={
            "size": getattr(processor, "size", None),
            "image_mean": getattr(processor, "image_mean", None),
            "image_std": getattr(processor, "image_std", None),
        },
    )

    return DatasetDict(splits), metadata


def write_preprocess_metadata(metadata: PreprocessMetadata, *, output_path: Path) -> Path:
    """Write preprocessing metadata and label maps to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
