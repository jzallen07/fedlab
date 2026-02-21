"""Run MedMNIST data quality and leakage audits."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.audit import audit_all_medmnist_datasets, audit_medmnist_dataset, write_audit_report
from src.ml.data import SUPPORTED_MEDMNIST_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MedMNIST dataset quality and leakage")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *SUPPORTED_MEDMNIST_DATASETS],
        help="Dataset to audit (default: all)",
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
        help="Image size passed to MedMNIST dataset constructors",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing MedMNIST files",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/data-audit/medmnist_audit_report.json",
        help="JSON report output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    if args.dataset == "all":
        reports = audit_all_medmnist_datasets(
            root=root,
            image_size=args.image_size,
            download=args.download,
        )
    else:
        reports = [
            audit_medmnist_dataset(
                args.dataset,
                root=root,
                image_size=args.image_size,
                download=args.download,
            )
        ]

    report_path = write_audit_report(reports, output_path=Path(args.report_path))
    print(f"Wrote audit report: {report_path}")
    for report in reports:
        leakage_flag = "yes" if report.leakage.has_cross_split_leakage else "no"
        print(
            f"- {report.dataset_id}: train={report.split_sizes['train']} "
            f"val={report.split_sizes['val']} test={report.split_sizes['test']} "
            f"cross_split_leakage={leakage_flag}"
        )


if __name__ == "__main__":
    main()
