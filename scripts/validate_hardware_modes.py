"""Validate CPU baseline and optional MPS fallback behavior for MedMNIST profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.data import SUPPORTED_MEDMNIST_DATASETS
from src.ml.hardware import (
    probe_mps_available,
    validate_all_hardware_modes,
    validate_dataset_hardware_modes,
    write_hardware_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CPU and MPS profile behavior")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *SUPPORTED_MEDMNIST_DATASETS],
        help="Dataset to validate (default: all)",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/hardware/hardware_validation_report.json",
        help="JSON report output path",
    )
    parser.add_argument(
        "--mps-available",
        default=None,
        choices=["true", "false"],
        help="Override runtime MPS probe (true/false). Defaults to actual probe.",
    )
    return parser.parse_args()


def _resolve_probe_override(raw: str | None) -> bool | None:
    if raw is None:
        return None
    return raw.lower() == "true"


def main() -> None:
    args = parse_args()
    override_probe = _resolve_probe_override(args.mps_available)
    observed_mps = probe_mps_available() if override_probe is None else override_probe

    if args.dataset == "all":
        results = validate_all_hardware_modes(probed_mps_available=observed_mps)
    else:
        results = [validate_dataset_hardware_modes(args.dataset, probed_mps_available=observed_mps)]

    report_path = write_hardware_validation_report(results, output_path=Path(args.report_path))
    print(f"Wrote hardware validation report: {report_path}")
    print(f"Runtime MPS available: {observed_mps}")
    for result in results:
        status = "pass" if result.validation_passed else "fail"
        print(
            f"- {result.dataset_id}: cpu={result.cpu_device} "
            f"mps_available={result.mps_device_when_available} "
            f"mps_unavailable={result.mps_device_when_unavailable} "
            f"auto={result.auto_device} status={status}"
        )


if __name__ == "__main__":
    main()
