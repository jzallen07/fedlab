"""Data quality and preprocessing audit utilities for MedMNIST datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from types import ModuleType

import numpy as np

from src.ml.data import DATASET_SPLITS, SUPPORTED_MEDMNIST_DATASETS


@dataclass(frozen=True)
class SplitAudit:
    """Audit summary for one dataset split."""

    split: str
    num_examples: int
    class_counts: dict[str, int]
    class_distribution: dict[str, float]
    integrity_ok: bool
    integrity_issues: list[str]
    duplicate_images_within_split: int


@dataclass(frozen=True)
class LeakageAudit:
    """Pairwise split overlap checks based on image fingerprints."""

    has_cross_split_leakage: bool
    pair_overlap_counts: dict[str, int]
    pair_overlap_rates: dict[str, float]


@dataclass(frozen=True)
class DatasetAuditReport:
    """Full data quality report for one MedMNIST dataset."""

    dataset_id: str
    root: str
    image_size: int
    loaded_image_size: int
    num_labels: int
    split_sizes: dict[str, int]
    splits: dict[str, SplitAudit]
    leakage: LeakageAudit


def _import_medmnist() -> ModuleType:
    try:
        import medmnist
    except ImportError as exc:
        raise RuntimeError(
            "medmnist is required for data audits. Install project dependencies first."
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


def _resolve_num_labels(medmnist: ModuleType, dataset_id: str) -> int:
    info = medmnist.INFO.get(dataset_id)
    if not isinstance(info, dict):
        raise ValueError(f"Dataset metadata not found in medmnist.INFO: {dataset_id}")
    labels = info.get("label")
    if not isinstance(labels, dict):
        raise ValueError(f"Dataset metadata missing label map for {dataset_id}")
    return len(labels)


def _class_balance(labels: np.ndarray) -> tuple[dict[str, int], dict[str, float]]:
    classes, counts = np.unique(labels, return_counts=True)
    total = int(labels.size)
    class_counts = {
        str(int(label)): int(count)
        for label, count in zip(classes, counts, strict=False)
    }
    if total == 0:
        return class_counts, {key: 0.0 for key in class_counts}
    distribution = {
        key: round(value / total, 6)
        for key, value in class_counts.items()
    }
    return class_counts, distribution


def _integrity_issues(*, labels: np.ndarray, num_labels: int, num_images: int) -> list[str]:
    issues: list[str] = []
    if labels.size == 0:
        issues.append("split is empty")
        return issues

    if labels.size != num_images:
        issues.append(
            f"labels/images mismatch: labels={labels.size}, images={num_images}"
        )
    if int(labels.min()) < 0:
        issues.append("negative label values detected")
    if int(labels.max()) >= num_labels:
        issues.append(
            f"label id exceeds label map: max_label={int(labels.max())}, num_labels={num_labels}"
        )
    return issues


def _fingerprint_image(image: np.ndarray) -> str:
    return hashlib.blake2b(np.asarray(image).tobytes(), digest_size=16).hexdigest()


def _split_hashes(images: np.ndarray) -> tuple[set[str], int]:
    hashes: set[str] = set()
    duplicate_count = 0
    for image in images:
        digest = _fingerprint_image(np.asarray(image))
        if digest in hashes:
            duplicate_count += 1
        else:
            hashes.add(digest)
    return hashes, duplicate_count


def _load_split_dataset(
    *,
    dataset_cls: type[object],
    split: str,
    root: Path,
    image_size: int,
    download: bool,
) -> tuple[object, int]:
    try:
        dataset = dataset_cls(
            split=split,
            root=str(root),
            size=image_size,
            download=download,
        )
        return dataset, image_size
    except RuntimeError as exc:
        if download or image_size == 28:
            raise
        if "Dataset not found" not in str(exc):
            raise
        dataset = dataset_cls(
            split=split,
            root=str(root),
            size=28,
            download=False,
        )
        return dataset, 28


def audit_medmnist_dataset(
    dataset_id: str,
    *,
    root: Path,
    image_size: int = 224,
    download: bool = False,
) -> DatasetAuditReport:
    """Audit one MedMNIST dataset for class balance, integrity, and leakage."""

    normalized_id = _normalize_dataset_id(dataset_id)
    medmnist = _import_medmnist()
    dataset_cls = _resolve_dataset_class(medmnist, normalized_id)
    num_labels = _resolve_num_labels(medmnist, normalized_id)

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    split_reports: dict[str, SplitAudit] = {}
    split_sizes: dict[str, int] = {}
    split_hash_sets: dict[str, set[str]] = {}
    loaded_image_size = image_size

    for split in DATASET_SPLITS:
        dataset, resolved_size = _load_split_dataset(
            dataset_cls=dataset_cls,
            split=split,
            root=root_path,
            image_size=image_size,
            download=download,
        )
        loaded_image_size = resolved_size
        labels = np.asarray(getattr(dataset, "labels", []), dtype=int).reshape(-1)
        images = np.asarray(getattr(dataset, "imgs", []))
        num_examples = int(labels.size)
        split_sizes[split] = num_examples

        class_counts, class_distribution = _class_balance(labels)
        issues = _integrity_issues(
            labels=labels,
            num_labels=num_labels,
            num_images=int(images.shape[0]),
        )
        hashes, duplicate_count = _split_hashes(images)
        split_hash_sets[split] = hashes

        split_reports[split] = SplitAudit(
            split=split,
            num_examples=num_examples,
            class_counts=class_counts,
            class_distribution=class_distribution,
            integrity_ok=not issues,
            integrity_issues=issues,
            duplicate_images_within_split=duplicate_count,
        )

    pair_overlap_counts: dict[str, int] = {}
    pair_overlap_rates: dict[str, float] = {}
    has_leakage = False
    for left, right in combinations(DATASET_SPLITS, 2):
        key = f"{left}__{right}"
        overlap_count = len(split_hash_sets[left] & split_hash_sets[right])
        min_size = min(split_sizes[left], split_sizes[right])
        overlap_rate = 0.0 if min_size == 0 else round(overlap_count / min_size, 6)
        pair_overlap_counts[key] = overlap_count
        pair_overlap_rates[key] = overlap_rate
        if overlap_count > 0:
            has_leakage = True

    return DatasetAuditReport(
        dataset_id=normalized_id,
        root=str(root_path),
        image_size=image_size,
        loaded_image_size=loaded_image_size,
        num_labels=num_labels,
        split_sizes=split_sizes,
        splits=split_reports,
        leakage=LeakageAudit(
            has_cross_split_leakage=has_leakage,
            pair_overlap_counts=pair_overlap_counts,
            pair_overlap_rates=pair_overlap_rates,
        ),
    )


def audit_all_medmnist_datasets(
    *,
    root: Path,
    image_size: int = 224,
    download: bool = False,
) -> list[DatasetAuditReport]:
    """Audit BloodMNIST, DermaMNIST, and PathMNIST datasets."""

    reports: list[DatasetAuditReport] = []
    for dataset_id in SUPPORTED_MEDMNIST_DATASETS:
        reports.append(
            audit_medmnist_dataset(
                dataset_id,
                root=root,
                image_size=image_size,
                download=download,
            )
        )
    return reports


def write_audit_report(reports: list[DatasetAuditReport], *, output_path: Path) -> Path:
    """Persist audit reports to a JSON artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"datasets": [asdict(report) for report in reports]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
