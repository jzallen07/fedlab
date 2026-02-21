"""Machine-learning modules for FedForge."""

from src.ml.audit import (
    DatasetAuditReport,
    LeakageAudit,
    SplitAudit,
    audit_all_medmnist_datasets,
    audit_medmnist_dataset,
    write_audit_report,
)
from src.ml.centralized import CentralizedTrainingReport, run_centralized_training
from src.ml.config import (
    ManifestValidationError,
    ResolvedRunConfig,
    resolve_run_config,
    validate_all_manifests,
)
from src.ml.data import (
    SUPPORTED_MEDMNIST_DATASETS,
    MedMNISTPreparationSummary,
    prepare_all_medmnist_datasets,
    prepare_medmnist_dataset,
    write_preparation_report,
)
from src.ml.hardware import (
    DatasetHardwareValidation,
    probe_mps_available,
    validate_all_hardware_modes,
    validate_dataset_hardware_modes,
    write_hardware_validation_report,
)
from src.ml.metrics import build_compute_metrics_fn, load_accuracy_metric
from src.ml.model import (
    MODEL_ID_DEIT_TINY,
    apply_train_mode,
    build_deit_model,
    count_trainable_parameters,
)
from src.ml.partition import (
    SUPPORTED_SKEW_PRESETS,
    PartitionResult,
    generate_partition_result,
    partition_labels_non_iid,
    partition_medmnist_dataset,
    write_partition_result,
)
from src.ml.preprocess import (
    PreprocessMetadata,
    build_image_processor,
    preprocess_medmnist_dataset,
    resolve_label_names,
    write_preprocess_metadata,
)
from src.ml.run_profile import (
    PhaseProfileReport,
    PhaseRunProfile,
    ProfileRunSummary,
    load_phase_run_profile,
    run_phase_profile,
    write_phase_profile_report,
)
from src.ml.trainer import TrainerRoundResult, create_trainer, run_trainer_round

__all__ = [
    "MODEL_ID_DEIT_TINY",
    "CentralizedTrainingReport",
    "DatasetAuditReport",
    "DatasetHardwareValidation",
    "LeakageAudit",
    "ManifestValidationError",
    "MedMNISTPreparationSummary",
    "PartitionResult",
    "PhaseProfileReport",
    "PhaseRunProfile",
    "PreprocessMetadata",
    "ProfileRunSummary",
    "ResolvedRunConfig",
    "SplitAudit",
    "SUPPORTED_MEDMNIST_DATASETS",
    "SUPPORTED_SKEW_PRESETS",
    "apply_train_mode",
    "audit_all_medmnist_datasets",
    "audit_medmnist_dataset",
    "build_compute_metrics_fn",
    "build_image_processor",
    "build_deit_model",
    "count_trainable_parameters",
    "create_trainer",
    "generate_partition_result",
    "load_accuracy_metric",
    "partition_labels_non_iid",
    "partition_medmnist_dataset",
    "prepare_all_medmnist_datasets",
    "prepare_medmnist_dataset",
    "preprocess_medmnist_dataset",
    "load_phase_run_profile",
    "probe_mps_available",
    "resolve_run_config",
    "resolve_label_names",
    "run_phase_profile",
    "run_centralized_training",
    "run_trainer_round",
    "TrainerRoundResult",
    "validate_all_hardware_modes",
    "validate_dataset_hardware_modes",
    "validate_all_manifests",
    "write_audit_report",
    "write_hardware_validation_report",
    "write_partition_result",
    "write_phase_profile_report",
    "write_preprocess_metadata",
    "write_preparation_report",
]
