"""Run centralized training on local MedMNIST data and emit model/loss artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.ml.centralized import run_centralized_training
from src.ml.data import SUPPORTED_MEDMNIST_DATASETS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run centralized MedMNIST training")
    parser.add_argument(
        "--dataset",
        default="bloodmnist",
        choices=list(SUPPORTED_MEDMNIST_DATASETS),
        help="Dataset id to train on",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run id (defaults to timestamped name)",
    )
    parser.add_argument(
        "--root",
        default="data/medmnist",
        help="Directory where MedMNIST files are stored",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/centralized",
        help="Root output directory",
    )
    parser.add_argument(
        "--model-id",
        default="hf-internal-testing/tiny-random-DeiTForImageClassification",
        help="HF model id or local model path used for central training",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "mps", "auto"],
        help="Requested device policy",
    )
    parser.add_argument(
        "--train-mode",
        default="head_only",
        choices=["head_only", "unfreeze_last_block"],
        help="Trainability policy",
    )
    parser.add_argument(
        "--image-size",
        default=28,
        type=int,
        help="MedMNIST source image size (must exist in local cache)",
    )
    parser.add_argument(
        "--epochs",
        default=3,
        type=int,
        help="Number of centralized training epochs",
    )
    parser.add_argument(
        "--logging-steps",
        default=50,
        type=int,
        help="Trainer logging interval in optimizer steps",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing dataset files",
    )
    parser.add_argument(
        "--max-train-samples",
        default=None,
        type=int,
        help="Optional cap on train split size",
    )
    parser.add_argument(
        "--max-val-samples",
        default=None,
        type=int,
        help="Optional cap on validation split size",
    )
    parser.add_argument(
        "--max-test-samples",
        default=None,
        type=int,
        help="Optional cap on test split size",
    )
    return parser.parse_args(argv)


def _default_run_id(dataset: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"central-{dataset}-{stamp}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or _default_run_id(args.dataset)
    report = run_centralized_training(
        run_id=run_id,
        dataset_id=args.dataset,
        root=Path(args.root),
        output_root=Path(args.output_dir),
        model_id=args.model_id,
        requested_device=args.device,
        train_mode=args.train_mode,
        image_size=args.image_size,
        download=args.download,
        epochs=args.epochs,
        logging_steps=args.logging_steps,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "dataset_id": report.dataset_id,
                "requested_device": report.requested_device,
                "resolved_device": report.resolved_device,
                "epochs": report.epochs,
                "train_examples": report.train_examples,
                "val_examples": report.val_examples,
                "test_examples": report.test_examples,
                "train_metrics": report.train_metrics,
                "val_metrics": report.val_metrics,
                "test_metrics": report.test_metrics,
                "artifacts": report.artifacts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

