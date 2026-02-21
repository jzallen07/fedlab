"""Prepare MedMNIST phase datasets for local FL runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.data import (
    SUPPORTED_MEDMNIST_DATASETS,
    prepare_all_medmnist_datasets,
    prepare_medmnist_dataset,
    write_preparation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MedMNIST datasets")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *SUPPORTED_MEDMNIST_DATASETS],
        help="Dataset to prepare (default: all)",
    )
    parser.add_argument(
        "--root",
        default="data/medmnist",
        help="Directory where MedMNIST files are stored",
    )
    parser.add_argument(
        "--image-size",
        default=224,
        type=int,
        help="Image size passed to MedMNIST dataset constructor",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download missing files",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/data/medmnist_preparation_report.json",
        help="JSON report output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    download = not args.skip_download
    if args.dataset == "all":
        summaries = prepare_all_medmnist_datasets(
            root=root,
            image_size=args.image_size,
            download=download,
        )
    else:
        summaries = [
            prepare_medmnist_dataset(
                args.dataset,
                root=root,
                image_size=args.image_size,
                download=download,
            )
        ]

    report_path = write_preparation_report(summaries, output_path=Path(args.report_path))
    print(f"Prepared {len(summaries)} dataset(s).")
    print(f"Wrote report: {report_path}")
    for summary in summaries:
        print(
            f"- {summary.dataset_id}: train={summary.split_sizes['train']} "
            f"val={summary.split_sizes['val']} test={summary.split_sizes['test']}"
        )


if __name__ == "__main__":
    main()
