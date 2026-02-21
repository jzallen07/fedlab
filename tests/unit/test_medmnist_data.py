from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ml.data import (
    SUPPORTED_MEDMNIST_DATASETS,
    prepare_all_medmnist_datasets,
    prepare_medmnist_dataset,
    write_preparation_report,
)


class _FakeDataset:
    split_sizes = {
        "train": 10,
        "val": 4,
        "test": 6,
    }
    calls: list[tuple[str, str, int, bool]] = []

    def __init__(self, *, split: str, root: str, size: int, download: bool) -> None:
        self.split = split
        self.root = root
        self.size = size
        self.download = download
        self.calls.append((split, root, size, download))

    def __len__(self) -> int:
        return self.split_sizes[self.split]


def _make_fake_medmnist() -> SimpleNamespace:
    info = {
        dataset_id: {
            "python_class": "FakeDataset",
            "label": {
                "0": "class_a",
                "1": "class_b",
            },
        }
        for dataset_id in SUPPORTED_MEDMNIST_DATASETS
    }
    return SimpleNamespace(INFO=info, FakeDataset=_FakeDataset)


def test_prepare_single_medmnist_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeDataset.calls = []
    monkeypatch.setattr("src.ml.data._import_medmnist", _make_fake_medmnist)

    summary = prepare_medmnist_dataset("bloodmnist", root=tmp_path, image_size=64, download=True)

    assert summary.dataset_id == "bloodmnist"
    assert summary.image_size == 64
    assert summary.split_sizes == {"train": 10, "val": 4, "test": 6}
    assert summary.num_labels == 2
    assert len(_FakeDataset.calls) == 3
    assert {split for split, *_ in _FakeDataset.calls} == {"train", "val", "test"}


def test_prepare_all_medmnist_datasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeDataset.calls = []
    monkeypatch.setattr("src.ml.data._import_medmnist", _make_fake_medmnist)

    summaries = prepare_all_medmnist_datasets(root=tmp_path, image_size=32, download=False)

    assert [summary.dataset_id for summary in summaries] == list(SUPPORTED_MEDMNIST_DATASETS)
    assert len(_FakeDataset.calls) == 9


def test_prepare_rejects_unknown_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.ml.data._import_medmnist", _make_fake_medmnist)

    with pytest.raises(ValueError):
        prepare_medmnist_dataset("unknown", root=tmp_path)


def test_write_preparation_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.ml.data._import_medmnist", _make_fake_medmnist)
    summaries = prepare_all_medmnist_datasets(root=tmp_path, image_size=28, download=False)

    output = write_preparation_report(summaries, output_path=tmp_path / "reports" / "prep.json")

    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "bloodmnist" in text
    assert "dermamnist" in text
    assert "pathmnist" in text
