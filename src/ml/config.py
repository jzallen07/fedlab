"""Training manifest loading and validation for phased MedMNIST runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

DeviceMode = Literal["auto", "cpu", "mps"]
ResolvedDevice = Literal["cpu", "mps"]
TrainMode = Literal["head_only", "unfreeze_last_block"]

DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "configs" / "training"
MANIFEST_FILES = {
    "bloodmnist": "bloodmnist.yaml",
    "dermamnist": "dermamnist.yaml",
    "pathmnist": "pathmnist.yaml",
}

_REQUIRED_TOP_LEVEL_KEYS = {
    "dataset",
    "phase",
    "run",
    "training",
    "device_profiles",
}
_REQUIRED_DATASET_KEYS = {"id", "name", "num_labels"}
_REQUIRED_RUN_KEYS = {"seed", "num_clients", "rounds", "default_train_mode"}
_REQUIRED_TRAINING_DEFAULTS = {
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "weight_decay",
    "warmup_ratio",
    "logging_steps",
    "eval_strategy",
    "save_strategy",
    "dataloader_num_workers",
    "max_grad_norm",
    "fp16",
    "bf16",
}
_REQUIRED_TRAINING_KEYS = {"defaults", "learning_rates", "mps_overrides"}
_REQUIRED_LEARNING_RATE_KEYS = {"head_only", "unfreeze_last_block"}
_REQUIRED_DEVICE_KEYS = {"cpu", "mps"}
_REQUIRED_MPS_KEYS = {"enabled_by_default", "fallback_to_cpu", "set_pytorch_mps_fallback"}


class ManifestValidationError(ValueError):
    """Raised when a training manifest is invalid."""


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Resolved training settings for a single run invocation."""

    dataset_id: str
    dataset_name: str
    phase: str
    run_seed: int
    num_clients: int
    rounds: int
    train_mode: TrainMode
    requested_device: DeviceMode
    device: ResolvedDevice
    fallback_applied: bool
    training_args: dict[str, Any]
    env: dict[str, str]


@dataclass(frozen=True)
class DatasetManifest:
    """A validated dataset manifest payload."""

    dataset_id: str
    dataset_name: str
    num_labels: int
    phase: str
    run: dict[str, Any]
    training: dict[str, Any]
    device_profiles: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestValidationError(f"Manifest not found: {path}") from exc

    if not isinstance(payload, dict):
        raise ManifestValidationError(f"Manifest is not a mapping: {path}")

    return payload


def _require_keys(name: str, payload: dict[str, Any], keys: set[str]) -> None:
    missing = sorted(keys - payload.keys())
    if missing:
        raise ManifestValidationError(f"{name} missing keys: {', '.join(missing)}")


def _validate_manifest(path: Path, payload: dict[str, Any]) -> DatasetManifest:
    _require_keys("manifest", payload, _REQUIRED_TOP_LEVEL_KEYS)

    dataset = payload["dataset"]
    run = payload["run"]
    training = payload["training"]
    profiles = payload["device_profiles"]

    if not isinstance(dataset, dict):
        raise ManifestValidationError(f"dataset section must be mapping: {path}")
    if not isinstance(run, dict):
        raise ManifestValidationError(f"run section must be mapping: {path}")
    if not isinstance(training, dict):
        raise ManifestValidationError(f"training section must be mapping: {path}")
    if not isinstance(profiles, dict):
        raise ManifestValidationError(f"device_profiles section must be mapping: {path}")

    _require_keys("dataset", dataset, _REQUIRED_DATASET_KEYS)
    _require_keys("run", run, _REQUIRED_RUN_KEYS)
    _require_keys("training", training, _REQUIRED_TRAINING_KEYS)
    _require_keys("training.defaults", training["defaults"], _REQUIRED_TRAINING_DEFAULTS)
    _require_keys(
        "training.learning_rates",
        training["learning_rates"],
        _REQUIRED_LEARNING_RATE_KEYS,
    )
    _require_keys("device_profiles", profiles, _REQUIRED_DEVICE_KEYS)
    _require_keys("device_profiles.mps", profiles["mps"], _REQUIRED_MPS_KEYS)

    if run["default_train_mode"] not in _REQUIRED_LEARNING_RATE_KEYS:
        raise ManifestValidationError(
            "run.default_train_mode must be head_only or unfreeze_last_block"
        )

    return DatasetManifest(
        dataset_id=str(dataset["id"]),
        dataset_name=str(dataset["name"]),
        num_labels=int(dataset["num_labels"]),
        phase=str(payload["phase"]),
        run=run,
        training=training,
        device_profiles=profiles,
    )


def list_manifest_paths(manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> list[Path]:
    """Return all configured manifest paths in deterministic order."""

    return [manifest_dir / name for name in MANIFEST_FILES.values()]


def load_manifest(dataset_id: str, manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> DatasetManifest:
    """Load and validate a dataset manifest by id."""

    dataset_key = dataset_id.strip().lower()
    if dataset_key not in MANIFEST_FILES:
        expected = ", ".join(sorted(MANIFEST_FILES))
        raise ManifestValidationError(
            f"Unknown dataset '{dataset_id}'. Expected one of: {expected}"
        )

    path = manifest_dir / MANIFEST_FILES[dataset_key]
    payload = _load_yaml(path)
    manifest = _validate_manifest(path, payload)

    if manifest.dataset_id != dataset_key:
        raise ManifestValidationError(
            "dataset.id mismatch in "
            f"{path.name}: expected '{dataset_key}', got '{manifest.dataset_id}'"
        )

    return manifest


def resolve_device_mode(
    requested_device: DeviceMode,
    *,
    mps_available: bool,
    mps_enabled: bool,
    fallback_to_cpu: bool,
) -> tuple[ResolvedDevice, bool]:
    """Resolve final device mode with optional mps fallback."""

    if requested_device not in {"auto", "cpu", "mps"}:
        raise ManifestValidationError(f"Invalid device mode: {requested_device}")

    if requested_device == "cpu":
        return "cpu", False

    wants_mps = requested_device == "mps" or requested_device == "auto"
    if wants_mps and mps_available and mps_enabled:
        return "mps", False

    if requested_device == "mps" and not fallback_to_cpu:
        raise ManifestValidationError(
            "MPS requested but unavailable/disabled and fallback_to_cpu is false"
        )

    return "cpu", requested_device == "mps"


def resolve_run_config(
    dataset_id: str,
    *,
    requested_device: DeviceMode = "auto",
    train_mode: TrainMode | None = None,
    mps_available: bool = False,
    enable_mps: bool | None = None,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
) -> ResolvedRunConfig:
    """Resolve runtime configuration for training based on manifest and device inputs."""

    manifest = load_manifest(dataset_id, manifest_dir=manifest_dir)

    selected_train_mode: TrainMode = train_mode or manifest.run["default_train_mode"]
    if selected_train_mode not in _REQUIRED_LEARNING_RATE_KEYS:
        raise ManifestValidationError(f"Unsupported train_mode: {selected_train_mode}")

    mps_profile = manifest.device_profiles["mps"]
    mps_enabled = (
        bool(mps_profile["enabled_by_default"]) if enable_mps is None else bool(enable_mps)
    )
    resolved_device, fallback_applied = resolve_device_mode(
        requested_device,
        mps_available=mps_available,
        mps_enabled=mps_enabled,
        fallback_to_cpu=bool(mps_profile["fallback_to_cpu"]),
    )

    training_args = dict(manifest.training["defaults"])
    training_args["learning_rate"] = manifest.training["learning_rates"][selected_train_mode]
    if resolved_device == "mps":
        training_args.update(manifest.training["mps_overrides"])

    env: dict[str, str] = {}
    if resolved_device == "mps" and bool(mps_profile["set_pytorch_mps_fallback"]):
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    return ResolvedRunConfig(
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        phase=manifest.phase,
        run_seed=int(manifest.run["seed"]),
        num_clients=int(manifest.run["num_clients"]),
        rounds=int(manifest.run["rounds"]),
        train_mode=selected_train_mode,
        requested_device=requested_device,
        device=resolved_device,
        fallback_applied=fallback_applied,
        training_args=training_args,
        env=env,
    )


def validate_all_manifests(manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> dict[str, DatasetManifest]:
    """Load and validate all known dataset manifests."""

    manifests: dict[str, DatasetManifest] = {}
    for dataset_id in sorted(MANIFEST_FILES):
        manifests[dataset_id] = load_manifest(dataset_id, manifest_dir=manifest_dir)
    return manifests
