"""Machine-learning modules for FedForge."""

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
from src.ml.trainer import TrainerRoundResult, create_trainer, run_trainer_round

__all__ = [
    "MODEL_ID_DEIT_TINY",
    "ManifestValidationError",
    "MedMNISTPreparationSummary",
    "PartitionResult",
    "PreprocessMetadata",
    "ResolvedRunConfig",
    "SUPPORTED_MEDMNIST_DATASETS",
    "SUPPORTED_SKEW_PRESETS",
    "apply_train_mode",
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
    "resolve_run_config",
    "resolve_label_names",
    "run_trainer_round",
    "TrainerRoundResult",
    "validate_all_manifests",
    "write_partition_result",
    "write_preprocess_metadata",
    "write_preparation_report",
]
