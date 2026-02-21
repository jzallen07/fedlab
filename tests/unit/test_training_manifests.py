from pathlib import Path

import pytest

from src.ml.config import (
    DEFAULT_MANIFEST_DIR,
    ManifestValidationError,
    list_manifest_paths,
    load_manifest,
    resolve_run_config,
    validate_all_manifests,
)


def test_manifest_files_exist() -> None:
    paths = list_manifest_paths()
    assert [path.name for path in paths] == [
        "bloodmnist.yaml",
        "dermamnist.yaml",
        "pathmnist.yaml",
    ]
    for path in paths:
        assert path.exists(), f"missing manifest: {path}"


def test_validate_all_manifests() -> None:
    manifests = validate_all_manifests()
    assert sorted(manifests.keys()) == ["bloodmnist", "dermamnist", "pathmnist"]
    assert manifests["bloodmnist"].dataset_name == "BloodMNIST"
    assert manifests["dermamnist"].dataset_name == "DermaMNIST"
    assert manifests["pathmnist"].dataset_name == "PathMNIST"


@pytest.mark.parametrize("dataset", ["bloodmnist", "dermamnist", "pathmnist"])
def test_cpu_profile_resolution(dataset: str) -> None:
    resolved = resolve_run_config(
        dataset,
        requested_device="cpu",
        mps_available=False,
        enable_mps=False,
    )

    assert resolved.device == "cpu"
    assert resolved.fallback_applied is False
    assert resolved.training_args["per_device_train_batch_size"] == 8
    assert resolved.training_args["per_device_eval_batch_size"] == 8
    assert resolved.training_args["fp16"] is False
    assert resolved.training_args["bf16"] is False


@pytest.mark.parametrize("dataset", ["bloodmnist", "dermamnist", "pathmnist"])
def test_optional_mps_profile_resolution(dataset: str) -> None:
    resolved = resolve_run_config(
        dataset,
        requested_device="mps",
        mps_available=True,
        enable_mps=True,
    )

    assert resolved.device == "mps"
    assert resolved.fallback_applied is False
    assert resolved.training_args["per_device_train_batch_size"] == 4
    assert resolved.training_args["per_device_eval_batch_size"] == 4
    assert resolved.env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


@pytest.mark.parametrize("dataset", ["bloodmnist", "dermamnist", "pathmnist"])
def test_mps_request_falls_back_to_cpu_when_unavailable(dataset: str) -> None:
    resolved = resolve_run_config(
        dataset,
        requested_device="mps",
        mps_available=False,
        enable_mps=True,
    )

    assert resolved.device == "cpu"
    assert resolved.fallback_applied is True
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in resolved.env


def test_unknown_dataset_raises_validation_error() -> None:
    with pytest.raises(ManifestValidationError):
        load_manifest("does-not-exist")


def test_custom_manifest_dir_support(tmp_path: Path) -> None:
    source_manifest = DEFAULT_MANIFEST_DIR / "bloodmnist.yaml"
    target_manifest = tmp_path / "bloodmnist.yaml"
    target_manifest.write_text(source_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ManifestValidationError):
        load_manifest("dermamnist", manifest_dir=tmp_path)
