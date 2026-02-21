"""Run a dataset phase profile and emit repeatability reference metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.run_profile import (
    load_phase_run_profile,
    run_phase_profile,
    write_phase_profile_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MedMNIST phase profile")
    parser.add_argument(
        "--profile",
        default="configs/profiles/bloodmnist_baseline.yaml",
        help="Path to phase profile YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/profiles",
        help="Directory where per-run summaries are written",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for profile report JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    profile = load_phase_run_profile(Path(args.profile))
    profile_dir = Path(args.output_dir) / profile.profile_id
    report = run_phase_profile(profile, output_dir=profile_dir)

    report_path = (
        Path(args.report_path)
        if args.report_path is not None
        else profile_dir / "reference_metrics.json"
    )
    write_phase_profile_report(report, output_path=report_path)

    print(f"Profile: {profile.profile_id}")
    print(f"Dataset: {report.dataset_id}")
    print(
        f"Baseline run: rounds={report.baseline_rounds} clients={report.baseline_clients} "
        f"batch_size={report.used_per_device_train_batch_size}"
    )
    print(
        f"Repeatability: repeatable={report.repeatable} "
        f"max_loss_delta={report.max_loss_delta_vs_run_1:.12f}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
