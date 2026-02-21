from pathlib import Path

import numpy as np
import pytest

from src.ml.partition import (
    generate_partition_result,
    partition_labels_non_iid,
    write_partition_result,
)


@pytest.mark.parametrize(
    ("dataset_id", "num_labels"),
    [
        ("bloodmnist", 8),
        ("dermamnist", 7),
        ("pathmnist", 9),
    ],
)
def test_partition_determinism_for_all_supported_datasets(
    dataset_id: str,
    num_labels: int,
) -> None:
    labels = np.arange(300) % num_labels

    first = generate_partition_result(
        dataset_id,
        labels=labels,
        split="train",
        num_clients=5,
        seed=1234,
        preset="moderate",
    )
    second = generate_partition_result(
        dataset_id,
        labels=labels,
        split="train",
        num_clients=5,
        seed=1234,
        preset="moderate",
    )

    assert first.client_indices == second.client_indices
    assert first.client_class_counts == second.client_class_counts


@pytest.mark.parametrize("preset", ["balanced", "mild", "moderate", "extreme"])
def test_partition_covers_all_samples_without_duplicates(preset: str) -> None:
    labels = np.asarray(([0] * 40) + ([1] * 35) + ([2] * 25))

    assignments, _ = partition_labels_non_iid(
        labels,
        num_clients=4,
        seed=555,
        preset=preset,  # type: ignore[arg-type]
    )

    all_indices = [index for values in assignments.values() for index in values]

    assert sorted(all_indices) == list(range(len(labels)))
    assert len(all_indices) == len(set(all_indices))
    assert all(len(values) > 0 for values in assignments.values())


def test_partition_seed_changes_output() -> None:
    labels = np.arange(240) % 6

    first, _ = partition_labels_non_iid(labels, num_clients=3, seed=1, preset="moderate")
    second, _ = partition_labels_non_iid(labels, num_clients=3, seed=2, preset="moderate")

    assert first != second


def test_partition_rejects_invalid_preset() -> None:
    labels = np.arange(30) % 3

    with pytest.raises(ValueError):
        partition_labels_non_iid(labels, num_clients=3, seed=11, preset="unknown")  # type: ignore[arg-type]


def test_write_partition_result(tmp_path: Path) -> None:
    labels = np.arange(90) % 3
    result = generate_partition_result(
        "bloodmnist",
        labels=labels,
        split="train",
        num_clients=3,
        seed=2026,
        preset="mild",
    )

    output = write_partition_result(result, output_path=tmp_path / "partitions" / "blood.json")

    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "client_indices" in text
    assert "bloodmnist" in text
