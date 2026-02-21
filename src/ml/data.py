"""MedMNIST dataset preparation helpers for Blood/Derma/Path phases."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

SUPPORTED_MEDMNIST_DATASETS: tuple[str, ...] = ("bloodmnist", "dermamnist", "pathmnist")
DATASET_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class MedMNISTPreparationSummary:
    """Prepared dataset metadata for one MedMNIST dataset."""

    dataset_id: str
    root: str
    image_size: int
    split_sizes: dict[str, int]
    num_labels: int
    labels: dict[str, str]


def _import_medmnist() -> ModuleType:
    try:
        import medmnist
    except ImportError as exc:
        raise RuntimeError(
            "medmnist is required for dataset preparation. Install project dependencies first."
        ) from exc
    return medmnist


def _normalize_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.strip().lower()
    if normalized not in SUPPORTED_MEDMNIST_DATASETS:
        expected = ", ".join(SUPPORTED_MEDMNIST_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_id}'. Expected one of: {expected}")
    return normalized


def _resolve_dataset_class(medmnist: ModuleType, dataset_id: str) -> type[Any]:
    info = medmnist.INFO.get(dataset_id)
    if info is None:
        raise ValueError(f"Dataset metadata not found in medmnist.INFO: {dataset_id}")

    class_name = info.get("python_class")
    if not isinstance(class_name, str):
        raise ValueError(f"Dataset metadata missing python_class for {dataset_id}")

    dataset_cls = getattr(medmnist, class_name, None)
    if dataset_cls is None:
        raise ValueError(f"Dataset class '{class_name}' not found in medmnist module")

    return dataset_cls


def _resolve_dataset_labels(medmnist: ModuleType, dataset_id: str) -> dict[str, str]:
    info = medmnist.INFO.get(dataset_id)
    labels = info.get("label") if info is not None else None
    if not isinstance(labels, dict):
        raise ValueError(f"Dataset metadata missing label map for {dataset_id}")
    return {str(key): str(value) for key, value in labels.items()}


def prepare_medmnist_dataset(
    dataset_id: str,
    *,
    root: Path,
    image_size: int = 224,
    download: bool = True,
) -> MedMNISTPreparationSummary:
    """Download/prepare a single MedMNIST dataset for all required splits."""

    normalized_id = _normalize_dataset_id(dataset_id)
    medmnist = _import_medmnist()
    dataset_cls = _resolve_dataset_class(medmnist, normalized_id)
    label_map = _resolve_dataset_labels(medmnist, normalized_id)

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    split_sizes: dict[str, int] = {}
    for split in DATASET_SPLITS:
        dataset = dataset_cls(split=split, root=str(root_path), size=image_size, download=download)
        split_sizes[split] = len(dataset)

    return MedMNISTPreparationSummary(
        dataset_id=normalized_id,
        root=str(root_path),
        image_size=image_size,
        split_sizes=split_sizes,
        num_labels=len(label_map),
        labels=label_map,
    )


def prepare_all_medmnist_datasets(
    *,
    root: Path,
    image_size: int = 224,
    download: bool = True,
) -> list[MedMNISTPreparationSummary]:
    """Prepare BloodMNIST, DermaMNIST, and PathMNIST datasets."""

    summaries: list[MedMNISTPreparationSummary] = []
    for dataset_id in SUPPORTED_MEDMNIST_DATASETS:
        summaries.append(
            prepare_medmnist_dataset(
                dataset_id,
                root=root,
                image_size=image_size,
                download=download,
            )
        )
    return summaries


def write_preparation_report(
    summaries: list[MedMNISTPreparationSummary],
    *,
    output_path: Path,
) -> Path:
    """Write dataset preparation results to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "datasets": [asdict(summary) for summary in summaries],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
