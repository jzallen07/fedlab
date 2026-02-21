"""HF Trainer orchestration for federated client local epochs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainingArguments

from src.ml.config import DeviceMode, ResolvedRunConfig, TrainMode, resolve_run_config
from src.ml.metrics import build_compute_metrics_fn
from src.ml.model import apply_train_mode, build_deit_model, count_trainable_parameters

_TRAINING_ARGS_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)


@dataclass(frozen=True)
class TrainerRoundResult:
    """Materialized outputs from one local trainer round."""

    resolved_config: ResolvedRunConfig
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float]
    trainable_parameters: int


def _set_if_supported(kwargs: dict[str, Any], name: str, value: Any) -> None:
    if name in _TRAINING_ARGS_PARAMS:
        kwargs[name] = value


def _filter_training_args_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in _TRAINING_ARGS_PARAMS}


def _to_float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            clean[key] = float(value)
    return clean


def build_training_arguments(
    resolved_config: ResolvedRunConfig,
    *,
    output_dir: Path,
    logging_dir: Path | None = None,
) -> TrainingArguments:
    """Create TrainingArguments from resolved run config in a version-safe way."""

    defaults = resolved_config.training_args
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": defaults["per_device_train_batch_size"],
        "per_device_eval_batch_size": defaults["per_device_eval_batch_size"],
        "gradient_accumulation_steps": defaults["gradient_accumulation_steps"],
        "num_train_epochs": defaults["num_train_epochs"],
        "learning_rate": defaults["learning_rate"],
        "weight_decay": defaults["weight_decay"],
        "warmup_ratio": defaults["warmup_ratio"],
        "logging_steps": defaults["logging_steps"],
        "save_strategy": defaults["save_strategy"],
        "dataloader_num_workers": defaults["dataloader_num_workers"],
        "max_grad_norm": defaults["max_grad_norm"],
        "seed": resolved_config.run_seed,
        "fp16": bool(defaults["fp16"]),
        "bf16": bool(defaults["bf16"]),
        "report_to": [],
        "disable_tqdm": True,
        "remove_unused_columns": False,
    }

    eval_strategy = defaults["eval_strategy"]
    if "eval_strategy" in _TRAINING_ARGS_PARAMS:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in _TRAINING_ARGS_PARAMS:
        kwargs["evaluation_strategy"] = eval_strategy

    if logging_dir is not None:
        kwargs["logging_dir"] = str(logging_dir)

    if resolved_config.device == "cpu":
        if "use_cpu" in _TRAINING_ARGS_PARAMS:
            kwargs["use_cpu"] = True
        else:
            _set_if_supported(kwargs, "no_cuda", True)
    elif resolved_config.device == "mps":
        _set_if_supported(kwargs, "use_mps_device", True)

    filtered = _filter_training_args_kwargs(kwargs)
    return TrainingArguments(**filtered)


def create_trainer(
    *,
    dataset_id: str,
    train_dataset: Any,
    eval_dataset: Any,
    num_labels: int,
    label_names: list[str] | None,
    output_dir: Path,
    requested_device: DeviceMode = "auto",
    train_mode: TrainMode | None = None,
    mps_available: bool = False,
    enable_mps: bool | None = None,
    model_id: str = "facebook/deit-tiny-patch16-224",
) -> tuple[Trainer, ResolvedRunConfig]:
    """Build a Trainer and resolved config for one client local epoch."""

    resolved = resolve_run_config(
        dataset_id,
        requested_device=requested_device,
        train_mode=train_mode,
        mps_available=mps_available,
        enable_mps=enable_mps,
    )

    model = build_deit_model(
        num_labels=num_labels,
        model_id=model_id,
        label_names=label_names,
    )
    apply_train_mode(model, resolved.train_mode)

    training_args = build_training_arguments(
        resolved,
        output_dir=output_dir,
        logging_dir=output_dir / "logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(),
    )

    return trainer, resolved


def run_trainer_round(
    *,
    dataset_id: str,
    train_dataset: Any,
    eval_dataset: Any,
    num_labels: int,
    label_names: list[str] | None,
    output_dir: Path,
    requested_device: DeviceMode = "auto",
    train_mode: TrainMode | None = None,
    mps_available: bool = False,
    enable_mps: bool | None = None,
    model_id: str = "facebook/deit-tiny-patch16-224",
) -> TrainerRoundResult:
    """Execute one local train+eval round and return metrics."""

    trainer, resolved = create_trainer(
        dataset_id=dataset_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_labels=num_labels,
        label_names=label_names,
        output_dir=output_dir,
        requested_device=requested_device,
        train_mode=train_mode,
        mps_available=mps_available,
        enable_mps=enable_mps,
        model_id=model_id,
    )

    train_output = trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)

    return TrainerRoundResult(
        resolved_config=resolved,
        train_metrics=_to_float_metrics(train_output.metrics),
        eval_metrics=_to_float_metrics(eval_metrics),
        trainable_parameters=count_trainable_parameters(trainer.model),
    )
