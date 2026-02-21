from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.ml.preprocess import (
    load_medmnist_split,
    preprocess_medmnist_dataset,
    resolve_label_names,
    write_preprocess_metadata,
)


class _FakeDataset:
    split_sizes = {
        "train": 6,
        "val": 4,
        "test": 3,
    }

    def __init__(self, *, split: str, root: str, size: int, download: bool) -> None:
        del root, download
        self.split = split
        self.size = size
        num_examples = self.split_sizes[split]
        self.labels = np.arange(num_examples) % 3

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        label = int(self.labels[index])
        image = np.full((self.size, self.size, 3), fill_value=label * 30, dtype=np.uint8)
        return image, np.asarray([label], dtype=np.int64)


class _FakeProcessor:
    size = {"height": 16, "width": 16}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __call__(self, *, images: list[np.ndarray], return_tensors: str) -> dict[str, np.ndarray]:
        assert return_tensors == "np"
        stacked = np.stack([
            np.transpose(np.asarray(image, dtype=np.float32) / 255.0, (2, 0, 1))
            for image in images
        ])
        return {"pixel_values": stacked}


def _fake_medmnist_module() -> SimpleNamespace:
    info = {
        "bloodmnist": {
            "python_class": "FakeDataset",
            "label": {"0": "normal", "1": "abnormal", "2": "artifact"},
        },
        "dermamnist": {
            "python_class": "FakeDataset",
            "label": {"0": "nevus", "1": "melanoma", "2": "keratosis"},
        },
        "pathmnist": {
            "python_class": "FakeDataset",
            "label": {"0": "type_a", "1": "type_b", "2": "type_c"},
        },
    }
    return SimpleNamespace(INFO=info, FakeDataset=_FakeDataset)


@pytest.mark.parametrize("dataset_id", ["bloodmnist", "dermamnist", "pathmnist"])
def test_resolve_label_names_for_all_datasets(
    dataset_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.ml.preprocess._import_medmnist", _fake_medmnist_module)

    labels = resolve_label_names(dataset_id)

    assert len(labels) == 3
    assert isinstance(labels[0], str)


def test_load_medmnist_split_returns_images_and_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.ml.preprocess._import_medmnist", _fake_medmnist_module)

    images, labels = load_medmnist_split(
        "bloodmnist",
        split="train",
        root=tmp_path,
        image_size=12,
        download=False,
    )

    assert len(images) == 6
    assert len(labels) == 6
    assert images[0].shape == (12, 12, 3)
    assert isinstance(labels[0], int)


@pytest.mark.parametrize("dataset_id", ["bloodmnist", "dermamnist", "pathmnist"])
def test_preprocess_medmnist_dataset_builds_hf_splits(
    dataset_id: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.ml.preprocess._import_medmnist", _fake_medmnist_module)

    dataset_dict, metadata = preprocess_medmnist_dataset(
        dataset_id,
        root=tmp_path,
        image_size=16,
        model_id="fake/model",
        download=False,
        image_processor=_FakeProcessor(),
    )

    assert sorted(dataset_dict.keys()) == ["test", "train", "val"]
    assert metadata.dataset_id == dataset_id
    assert metadata.split_sizes == {"train": 6, "val": 4, "test": 3}

    sample = dataset_dict["train"][0]
    assert sample["pixel_values"].shape == (3, 16, 16)
    assert isinstance(sample["labels"].item(), (int, np.integer))


def test_write_preprocess_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.ml.preprocess._import_medmnist", _fake_medmnist_module)

    _, metadata = preprocess_medmnist_dataset(
        "bloodmnist",
        root=tmp_path,
        image_size=10,
        model_id="fake/model",
        download=False,
        image_processor=_FakeProcessor(),
    )

    output = write_preprocess_metadata(metadata, output_path=tmp_path / "metadata" / "blood.json")

    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "bloodmnist" in text
    assert "label2id" in text
