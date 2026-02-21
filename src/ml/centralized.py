"""Centralized (non-federated) training pipeline on local MedMNIST data."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from src.ml.data import SUPPORTED_MEDMNIST_DATASETS
from src.ml.hardware import probe_mps_available
from src.ml.preprocess import build_image_processor
from src.ml.trainer import create_trainer


def _import_medmnist() -> ModuleType:
    try:
        import medmnist
    except ImportError as exc:
        raise RuntimeError(
            "medmnist is required for centralized training. Install project dependencies first."
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


def _resolve_label_names(medmnist: ModuleType, dataset_id: str) -> list[str]:
    info = medmnist.INFO.get(dataset_id)
    if not isinstance(info, dict):
        raise ValueError(f"Dataset metadata not found in medmnist.INFO: {dataset_id}")

    raw_labels = info.get("label")
    if not isinstance(raw_labels, dict):
        raise ValueError(f"Dataset metadata missing label map for {dataset_id}")

    ordered_items = sorted(raw_labels.items(), key=lambda item: int(item[0]))
    return [str(label_name) for _, label_name in ordered_items]


class _HFMedMNISTDataset(torch.utils.data.Dataset):
    """Torch dataset wrapper that emits HF Trainer-compatible records."""

    def __init__(self, dataset: Any, image_processor: Any) -> None:
        self.dataset = dataset
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.dataset[index]
        encoded = self.image_processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        label_value = int(np.asarray(label).reshape(-1)[0])
        return {"pixel_values": pixel_values, "labels": label_value}


@dataclass(frozen=True)
class CurvePoint:
    step: int
    epoch: float | None
    loss: float
    accuracy: float | None = None


@dataclass(frozen=True)
class CentralizedTrainingReport:
    run_id: str
    dataset_id: str
    model_id: str
    requested_device: str
    resolved_device: str
    train_mode: str
    epochs: int
    train_examples: int
    val_examples: int
    test_examples: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    artifacts: dict[str, str]


def _as_float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _subset_dataset(
    dataset: torch.utils.data.Dataset,
    limit: int | None,
) -> torch.utils.data.Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("Sample limits must be >= 1")
    return torch.utils.data.Subset(dataset, list(range(limit)))


def _extract_curves(log_history: list[dict[str, Any]]) -> tuple[list[CurvePoint], list[CurvePoint]]:
    train_points: list[CurvePoint] = []
    eval_points: list[CurvePoint] = []

    for entry in log_history:
        step_raw = entry.get("step")
        epoch_raw = entry.get("epoch")
        step = int(step_raw) if isinstance(step_raw, (int, float)) else 0
        epoch = float(epoch_raw) if isinstance(epoch_raw, (int, float)) else None

        if "loss" in entry and "eval_loss" not in entry and isinstance(entry["loss"], (int, float)):
            train_points.append(CurvePoint(step=step, epoch=epoch, loss=float(entry["loss"])))

        if "eval_loss" in entry and isinstance(entry["eval_loss"], (int, float)):
            accuracy: float | None = None
            if isinstance(entry.get("eval_accuracy"), (int, float)):
                accuracy = float(entry["eval_accuracy"])
            eval_points.append(
                CurvePoint(
                    step=step,
                    epoch=epoch,
                    loss=float(entry["eval_loss"]),
                    accuracy=accuracy,
                )
            )

    return train_points, eval_points


def _write_curve_csv(*, path: Path, points: list[CurvePoint], include_accuracy: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if include_accuracy:
            writer.writerow(["step", "epoch", "loss", "accuracy"])
            for point in points:
                writer.writerow([point.step, point.epoch, point.loss, point.accuracy])
        else:
            writer.writerow(["step", "epoch", "loss"])
            for point in points:
                writer.writerow([point.step, point.epoch, point.loss])
    return path


def _scale(
    value: float,
    domain_min: float,
    domain_max: float,
    range_min: float,
    range_max: float,
) -> float:
    if domain_max <= domain_min:
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def _polyline(
    points: list[CurvePoint],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    width = 900
    height = 420
    left = 56
    right = width - 24
    top = 24
    bottom = height - 44

    coords: list[str] = []
    for point in points:
        x = _scale(float(point.step), x_min, x_max, left, right)
        y = _scale(point.loss, y_min, y_max, bottom, top)
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def _write_loss_curve_svg(
    *,
    train_points: list[CurvePoint],
    eval_points: list[CurvePoint],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    all_points = train_points + eval_points
    if not all_points:
        path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='420'/>",
            encoding="utf-8",
        )
        return path

    x_values = [float(point.step) for point in all_points]
    y_values = [point.loss for point in all_points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    y_pad = max((y_max - y_min) * 0.05, 1e-6)
    y_min -= y_pad
    y_max += y_pad

    train_line = _polyline(train_points, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    eval_line = _polyline(eval_points, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420"
  viewBox="0 0 900 420">
  <rect x="0" y="0" width="900" height="420" fill="#ffffff" />
  <line x1="56" y1="376" x2="876" y2="376" stroke="#94a3b8" stroke-width="1.5" />
  <line x1="56" y1="24" x2="56" y2="376" stroke="#94a3b8" stroke-width="1.5" />
  <text x="56" y="400" font-family="monospace" font-size="12" fill="#334155">step</text>
  <text x="12" y="24" font-family="monospace" font-size="12" fill="#334155">loss</text>
  <text x="56" y="18" font-family="monospace" font-size="13" fill="#0f172a">
    Centralized Training Loss Curves
  </text>
  <polyline points="{train_line}" fill="none" stroke="#0ea5e9" stroke-width="2.0" />
  <polyline points="{eval_line}" fill="none" stroke="#f97316" stroke-width="2.0" />
  <rect x="640" y="16" width="12" height="4" fill="#0ea5e9" />
  <text x="658" y="22" font-family="monospace" font-size="12" fill="#0f172a">train loss</text>
  <rect x="760" y="16" width="12" height="4" fill="#f97316" />
  <text x="778" y="22" font-family="monospace" font-size="12" fill="#0f172a">eval loss</text>
  <text x="56" y="40" font-family="monospace" font-size="11" fill="#475569">
    min={y_min:.6f} max={y_max:.6f}
  </text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def run_centralized_training(
    *,
    run_id: str,
    dataset_id: str,
    root: Path,
    output_root: Path,
    model_id: str,
    requested_device: str,
    train_mode: str,
    image_size: int,
    download: bool,
    epochs: int,
    logging_steps: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
) -> CentralizedTrainingReport:
    """Run centralized training on local MedMNIST splits and save artifacts."""

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if logging_steps < 1:
        raise ValueError("logging_steps must be >= 1")
    if requested_device not in {"cpu", "mps", "auto"}:
        raise ValueError("requested_device must be one of: cpu, mps, auto")
    if train_mode not in {"head_only", "unfreeze_last_block"}:
        raise ValueError("train_mode must be one of: head_only, unfreeze_last_block")

    normalized_dataset = _normalize_dataset_id(dataset_id)
    output_dir = output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    medmnist = _import_medmnist()
    dataset_cls = _resolve_dataset_class(medmnist, normalized_dataset)
    label_names = _resolve_label_names(medmnist, normalized_dataset)
    num_labels = len(label_names)

    image_processor = build_image_processor(model_id=model_id)

    train_raw = dataset_cls(split="train", root=str(root), size=image_size, download=download)
    val_raw = dataset_cls(split="val", root=str(root), size=image_size, download=download)
    test_raw = dataset_cls(split="test", root=str(root), size=image_size, download=download)

    train_dataset = _subset_dataset(
        _HFMedMNISTDataset(train_raw, image_processor),
        max_train_samples,
    )
    val_dataset = _subset_dataset(
        _HFMedMNISTDataset(val_raw, image_processor),
        max_val_samples,
    )
    test_dataset = _subset_dataset(
        _HFMedMNISTDataset(test_raw, image_processor),
        max_test_samples,
    )

    mps_available = probe_mps_available()
    enable_mps = requested_device in {"auto", "mps"}
    trainer, resolved = create_trainer(
        dataset_id=normalized_dataset,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        num_labels=num_labels,
        label_names=label_names,
        output_dir=output_dir / "trainer",
        requested_device=requested_device,  # type: ignore[arg-type]
        train_mode=train_mode,  # type: ignore[arg-type]
        mps_available=mps_available,
        enable_mps=enable_mps,
        model_id=model_id,
    )

    trainer.args.num_train_epochs = float(epochs)
    trainer.args.logging_steps = int(logging_steps)
    trainer.args.save_strategy = "epoch"
    if hasattr(trainer.args, "eval_strategy"):
        trainer.args.eval_strategy = "epoch"  # type: ignore[attr-defined]
    if hasattr(trainer.args, "evaluation_strategy"):
        trainer.args.evaluation_strategy = "epoch"  # type: ignore[attr-defined]
    if hasattr(trainer.args, "save_total_limit"):
        trainer.args.save_total_limit = 3  # type: ignore[attr-defined]
    if hasattr(trainer.args, "load_best_model_at_end"):
        trainer.args.load_best_model_at_end = True  # type: ignore[attr-defined]
    if hasattr(trainer.args, "metric_for_best_model"):
        trainer.args.metric_for_best_model = "eval_loss"  # type: ignore[attr-defined]
    if hasattr(trainer.args, "greater_is_better"):
        trainer.args.greater_is_better = False  # type: ignore[attr-defined]

    train_output = trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))

    state_path = output_dir / "trainer_state.json"
    trainer.state.save_to_json(str(state_path))

    log_history_path = output_dir / "log_history.json"
    log_history_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    train_curve, eval_curve = _extract_curves(trainer.state.log_history)
    train_curve_csv = _write_curve_csv(
        path=output_dir / "train_loss_curve.csv",
        points=train_curve,
        include_accuracy=False,
    )
    eval_curve_csv = _write_curve_csv(
        path=output_dir / "eval_loss_curve.csv",
        points=eval_curve,
        include_accuracy=True,
    )
    loss_curve_svg = _write_loss_curve_svg(
        train_points=train_curve,
        eval_points=eval_curve,
        path=output_dir / "loss_curve.svg",
    )

    report = CentralizedTrainingReport(
        run_id=run_id,
        dataset_id=normalized_dataset,
        model_id=model_id,
        requested_device=requested_device,
        resolved_device=resolved.device,
        train_mode=train_mode,
        epochs=epochs,
        train_examples=len(train_dataset),
        val_examples=len(val_dataset),
        test_examples=len(test_dataset),
        train_metrics=_as_float_metrics(train_output.metrics),
        val_metrics=_as_float_metrics(val_metrics),
        test_metrics=_as_float_metrics(test_metrics),
        artifacts={
            "output_dir": str(output_dir),
            "model_dir": str(model_dir),
            "trainer_state": str(state_path),
            "log_history": str(log_history_path),
            "train_loss_curve_csv": str(train_curve_csv),
            "eval_loss_curve_csv": str(eval_curve_csv),
            "loss_curve_svg": str(loss_curve_svg),
        },
    )

    summary_path = output_dir / "summary.json"
    payload = asdict(report)
    payload["generated_at_utc"] = datetime.now(UTC).isoformat()
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return report
