"""Hardware-mode validation helpers for CPU baseline and optional MPS fallback."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ml.config import resolve_run_config
from src.ml.data import SUPPORTED_MEDMNIST_DATASETS


@dataclass(frozen=True)
class DatasetHardwareValidation:
    """Hardware validation result for one MedMNIST dataset profile."""

    dataset_id: str
    cpu_device: str
    cpu_repeatable: bool
    mps_device_when_available: str
    mps_device_when_unavailable: str
    mps_fallback_applied_when_unavailable: bool
    auto_device: str
    probed_mps_available: bool
    fallback_env_set_when_mps: bool
    validation_passed: bool


def probe_mps_available() -> bool:
    """Detect runtime MPS availability if torch is installed."""

    try:
        import torch
    except Exception:
        return False

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_available())


def _cpu_signature(dataset_id: str) -> tuple[str, int, dict[str, object]]:
    resolved = resolve_run_config(
        dataset_id,
        requested_device="cpu",
        mps_available=False,
        enable_mps=False,
    )
    return resolved.device, resolved.run_seed, resolved.training_args


def validate_dataset_hardware_modes(
    dataset_id: str,
    *,
    probed_mps_available: bool | None = None,
) -> DatasetHardwareValidation:
    """Validate CPU determinism and MPS fallback behavior for one dataset."""

    observed_mps = probe_mps_available() if probed_mps_available is None else probed_mps_available

    cpu_one_device, cpu_one_seed, cpu_one_args = _cpu_signature(dataset_id)
    cpu_two_device, cpu_two_seed, cpu_two_args = _cpu_signature(dataset_id)
    cpu_repeatable = (
        cpu_one_device == cpu_two_device
        and cpu_one_seed == cpu_two_seed
        and cpu_one_args == cpu_two_args
    )

    mps_available = resolve_run_config(
        dataset_id,
        requested_device="mps",
        mps_available=True,
        enable_mps=True,
    )
    mps_unavailable = resolve_run_config(
        dataset_id,
        requested_device="mps",
        mps_available=False,
        enable_mps=True,
    )
    auto_selected = resolve_run_config(
        dataset_id,
        requested_device="auto",
        mps_available=observed_mps,
        enable_mps=True,
    )

    fallback_env_set = mps_available.env.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    validation_passed = all(
        [
            cpu_one_device == "cpu",
            cpu_repeatable,
            mps_available.device == "mps",
            mps_unavailable.device == "cpu",
            mps_unavailable.fallback_applied,
            fallback_env_set,
            auto_selected.device == ("mps" if observed_mps else "cpu"),
        ]
    )

    return DatasetHardwareValidation(
        dataset_id=dataset_id,
        cpu_device=cpu_one_device,
        cpu_repeatable=cpu_repeatable,
        mps_device_when_available=mps_available.device,
        mps_device_when_unavailable=mps_unavailable.device,
        mps_fallback_applied_when_unavailable=mps_unavailable.fallback_applied,
        auto_device=auto_selected.device,
        probed_mps_available=observed_mps,
        fallback_env_set_when_mps=fallback_env_set,
        validation_passed=validation_passed,
    )


def validate_all_hardware_modes(
    *,
    probed_mps_available: bool | None = None,
) -> list[DatasetHardwareValidation]:
    """Validate hardware behavior for all phased MedMNIST datasets."""

    results: list[DatasetHardwareValidation] = []
    for dataset_id in SUPPORTED_MEDMNIST_DATASETS:
        results.append(
            validate_dataset_hardware_modes(
                dataset_id,
                probed_mps_available=probed_mps_available,
            )
        )
    return results


def write_hardware_validation_report(
    results: list[DatasetHardwareValidation],
    *,
    output_path: Path,
) -> Path:
    """Write hardware validation artifact to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"datasets": [asdict(result) for result in results]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
