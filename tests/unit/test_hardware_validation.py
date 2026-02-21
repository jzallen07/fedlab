from pathlib import Path

import pytest

from src.ml.hardware import (
    validate_dataset_hardware_modes,
    write_hardware_validation_report,
)


@pytest.mark.parametrize("dataset_id", ["bloodmnist", "dermamnist", "pathmnist"])
def test_validate_dataset_hardware_modes_cpu_fallback(dataset_id: str) -> None:
    result = validate_dataset_hardware_modes(
        dataset_id,
        probed_mps_available=False,
    )

    assert result.dataset_id == dataset_id
    assert result.cpu_device == "cpu"
    assert result.cpu_repeatable is True
    assert result.mps_device_when_available == "mps"
    assert result.mps_device_when_unavailable == "cpu"
    assert result.mps_fallback_applied_when_unavailable is True
    assert result.auto_device == "cpu"
    assert result.fallback_env_set_when_mps is True
    assert result.validation_passed is True


def test_validate_dataset_hardware_modes_auto_selects_mps_when_available() -> None:
    result = validate_dataset_hardware_modes(
        "bloodmnist",
        probed_mps_available=True,
    )

    assert result.auto_device == "mps"
    assert result.validation_passed is True


def test_write_hardware_validation_report(tmp_path: Path) -> None:
    result = validate_dataset_hardware_modes(
        "dermamnist",
        probed_mps_available=False,
    )
    output_path = write_hardware_validation_report(
        [result],
        output_path=tmp_path / "hardware" / "report.json",
    )

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "dermamnist" in text
    assert "validation_passed" in text
