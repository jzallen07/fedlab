"""Deterministic non-IID partitioning utilities for MedMNIST client shards."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np

from src.ml.data import SUPPORTED_MEDMNIST_DATASETS

SkewPreset = Literal["balanced", "mild", "moderate", "extreme"]
SUPPORTED_SKEW_PRESETS: tuple[SkewPreset, ...] = (
    "balanced",
    "mild",
    "moderate",
    "extreme",
)
_DIRICHLET_ALPHA_BY_PRESET: dict[SkewPreset, float] = {
    "balanced": 1.0,
    "mild": 2.0,
    "moderate": 0.5,
    "extreme": 0.1,
}


@dataclass(frozen=True)
class PartitionResult:
    """Partition output for one dataset and one split."""

    dataset_id: str
    split: str
    seed: int
    preset: SkewPreset
    num_clients: int
    total_examples: int
    client_indices: dict[str, list[int]]
    client_class_counts: dict[str, dict[str, int]]


def _import_medmnist() -> ModuleType:
    try:
        import medmnist
    except ImportError as exc:
        raise RuntimeError(
            "medmnist is required for partition generation. Install project dependencies first."
        ) from exc
    return medmnist


def _normalize_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.strip().lower()
    if normalized not in SUPPORTED_MEDMNIST_DATASETS:
        expected = ", ".join(SUPPORTED_MEDMNIST_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_id}'. Expected one of: {expected}")
    return normalized


def _resolve_dataset_class(medmnist: ModuleType, dataset_id: str) -> type[object]:
    info = medmnist.INFO.get(dataset_id)
    if not isinstance(info, dict):
        raise ValueError(f"Dataset metadata not found in medmnist.INFO: {dataset_id}")

    class_name = info.get("python_class")
    if not isinstance(class_name, str):
        raise ValueError(f"Dataset metadata missing python_class for {dataset_id}")

    dataset_cls = getattr(medmnist, class_name, None)
    if dataset_cls is None:
        raise ValueError(f"Dataset class '{class_name}' not found in medmnist module")
    return dataset_cls


def _load_split_labels(
    dataset_id: str,
    *,
    split: str,
    root: Path,
    image_size: int,
    download: bool,
) -> np.ndarray:
    medmnist = _import_medmnist()
    dataset_cls = _resolve_dataset_class(medmnist, dataset_id)
    dataset = dataset_cls(split=split, root=str(root), size=image_size, download=download)

    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise ValueError(f"Dataset '{dataset_id}' does not expose .labels")

    labels_array = np.asarray(labels).reshape(-1)
    if labels_array.size == 0:
        raise ValueError(f"Dataset '{dataset_id}' split '{split}' has no labels")
    return labels_array.astype(int)


def _ensure_non_empty_clients(assignments: dict[str, list[int]], rng: np.random.Generator) -> None:
    empty_clients = [client_id for client_id, indices in assignments.items() if not indices]
    if not empty_clients:
        return

    for empty_client in empty_clients:
        donor = max(assignments, key=lambda client_id: len(assignments[client_id]))
        donor_indices = assignments[donor]
        if len(donor_indices) <= 1:
            raise ValueError("Cannot rebalance partition: insufficient donor samples")
        selected = int(rng.integers(0, len(donor_indices)))
        sample_index = donor_indices.pop(selected)
        assignments[empty_client].append(sample_index)


def _build_class_counts(
    *,
    labels: np.ndarray,
    assignments: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for client_id, indices in assignments.items():
        if not indices:
            counts[client_id] = {}
            continue

        client_labels = labels[np.asarray(indices)]
        classes, freqs = np.unique(client_labels, return_counts=True)
        counts[client_id] = {
            str(int(label)): int(freq)
            for label, freq in zip(classes, freqs, strict=False)
        }
    return counts


def partition_labels_non_iid(
    labels: np.ndarray,
    *,
    num_clients: int,
    seed: int,
    preset: SkewPreset,
) -> tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
    """Partition labels across clients with deterministic non-IID class skew."""

    if preset not in SUPPORTED_SKEW_PRESETS:
        expected = ", ".join(SUPPORTED_SKEW_PRESETS)
        raise ValueError(f"Unsupported skew preset '{preset}'. Expected one of: {expected}")
    if num_clients < 2:
        raise ValueError("num_clients must be >= 2")

    labels = np.asarray(labels).reshape(-1)
    if labels.size < num_clients:
        raise ValueError("Number of labels must be >= num_clients")

    rng = np.random.default_rng(seed)
    assignments: dict[str, list[int]] = {
        f"client_{client_idx}": []
        for client_idx in range(num_clients)
    }

    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        shuffled = rng.permutation(class_indices)

        if preset == "balanced":
            for offset, index in enumerate(shuffled):
                client_id = f"client_{offset % num_clients}"
                assignments[client_id].append(int(index))
            continue

        alpha = _DIRICHLET_ALPHA_BY_PRESET[preset]
        probabilities = rng.dirichlet(np.full(num_clients, alpha))
        class_counts = rng.multinomial(len(shuffled), probabilities)

        cursor = 0
        for client_idx, count in enumerate(class_counts):
            if count == 0:
                continue
            next_cursor = cursor + count
            client_id = f"client_{client_idx}"
            assignments[client_id].extend(int(idx) for idx in shuffled[cursor:next_cursor])
            cursor = next_cursor

    _ensure_non_empty_clients(assignments, rng)

    for indices in assignments.values():
        indices.sort()

    class_counts = _build_class_counts(labels=labels, assignments=assignments)
    return assignments, class_counts


def generate_partition_result(
    dataset_id: str,
    *,
    labels: np.ndarray,
    split: str,
    num_clients: int,
    seed: int,
    preset: SkewPreset,
) -> PartitionResult:
    """Generate deterministic partition result from provided labels."""

    normalized_id = _normalize_dataset_id(dataset_id)
    assignments, class_counts = partition_labels_non_iid(
        labels,
        num_clients=num_clients,
        seed=seed,
        preset=preset,
    )

    return PartitionResult(
        dataset_id=normalized_id,
        split=split,
        seed=seed,
        preset=preset,
        num_clients=num_clients,
        total_examples=int(np.asarray(labels).size),
        client_indices=assignments,
        client_class_counts=class_counts,
    )


def partition_medmnist_dataset(
    dataset_id: str,
    *,
    root: Path,
    split: str = "train",
    image_size: int = 224,
    download: bool = False,
    num_clients: int = 3,
    seed: int = 20260221,
    preset: SkewPreset = "moderate",
) -> PartitionResult:
    """Load MedMNIST labels and partition one split for federated clients."""

    normalized_id = _normalize_dataset_id(dataset_id)
    labels = _load_split_labels(
        normalized_id,
        split=split,
        root=root,
        image_size=image_size,
        download=download,
    )

    return generate_partition_result(
        normalized_id,
        labels=labels,
        split=split,
        num_clients=num_clients,
        seed=seed,
        preset=preset,
    )


def write_partition_result(result: PartitionResult, *, output_path: Path) -> Path:
    """Write one partition artifact to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
