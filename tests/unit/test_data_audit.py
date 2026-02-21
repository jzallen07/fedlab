from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.ml.audit import audit_medmnist_dataset, write_audit_report


def _image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)


class _LeakageDataset:
    split_payloads = {
        "train": {
            "labels": np.asarray([0, 1, 1], dtype=np.int64),
            "imgs": np.stack([_image(1), _image(2), _image(3)]),
        },
        "val": {
            "labels": np.asarray([1, 0], dtype=np.int64),
            "imgs": np.stack([_image(3), _image(4)]),
        },
        "test": {
            "labels": np.asarray([0, 1], dtype=np.int64),
            "imgs": np.stack([_image(5), _image(6)]),
        },
    }

    def __init__(self, *, split: str, root: str, size: int, download: bool) -> None:
        del root, size, download
        payload = self.split_payloads[split]
        self.labels = payload["labels"]
        self.imgs = payload["imgs"]

    def __len__(self) -> int:
        return int(self.labels.size)


class _BadLabelDataset:
    split_payloads = {
        "train": {
            "labels": np.asarray([0, 1], dtype=np.int64),
            "imgs": np.stack([_image(11), _image(12)]),
        },
        "val": {
            "labels": np.asarray([2], dtype=np.int64),
            "imgs": np.stack([_image(13)]),
        },
        "test": {
            "labels": np.asarray([0], dtype=np.int64),
            "imgs": np.stack([_image(14)]),
        },
    }

    def __init__(self, *, split: str, root: str, size: int, download: bool) -> None:
        del root, size, download
        payload = self.split_payloads[split]
        self.labels = payload["labels"]
        self.imgs = payload["imgs"]

    def __len__(self) -> int:
        return int(self.labels.size)


def _fake_medmnist_module(dataset_cls: type[object]) -> SimpleNamespace:
    info = {
        "bloodmnist": {
            "python_class": "FakeDataset",
            "label": {"0": "normal", "1": "abnormal"},
        },
        "dermamnist": {
            "python_class": "FakeDataset",
            "label": {"0": "nevus", "1": "melanoma"},
        },
        "pathmnist": {
            "python_class": "FakeDataset",
            "label": {"0": "type_a", "1": "type_b"},
        },
    }
    return SimpleNamespace(INFO=info, FakeDataset=dataset_cls)


def test_audit_medmnist_dataset_reports_class_balance_and_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.ml.audit._import_medmnist",
        lambda: _fake_medmnist_module(_LeakageDataset),
    )

    report = audit_medmnist_dataset("bloodmnist", root=tmp_path, image_size=8, download=False)

    assert report.dataset_id == "bloodmnist"
    assert report.split_sizes == {"train": 3, "val": 2, "test": 2}
    assert report.splits["train"].class_counts == {"0": 1, "1": 2}
    assert report.splits["train"].integrity_ok is True
    assert report.leakage.has_cross_split_leakage is True
    assert report.leakage.pair_overlap_counts["train__val"] == 1
    assert report.leakage.pair_overlap_counts["train__test"] == 0


def test_audit_medmnist_dataset_flags_label_integrity_issues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.ml.audit._import_medmnist",
        lambda: _fake_medmnist_module(_BadLabelDataset),
    )

    report = audit_medmnist_dataset("bloodmnist", root=tmp_path, image_size=8, download=False)

    assert report.splits["val"].integrity_ok is False
    assert any(
        "label id exceeds label map" in issue
        for issue in report.splits["val"].integrity_issues
    )


def test_write_audit_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.ml.audit._import_medmnist",
        lambda: _fake_medmnist_module(_LeakageDataset),
    )
    report = audit_medmnist_dataset("dermamnist", root=tmp_path, image_size=8, download=False)

    output_path = write_audit_report([report], output_path=tmp_path / "audit" / "report.json")

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "dermamnist" in text
    assert "pair_overlap_counts" in text
