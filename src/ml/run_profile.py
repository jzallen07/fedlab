"""Dataset phase run profiles and baseline execution helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.ml.config import MANIFEST_FILES, resolve_run_config

if TYPE_CHECKING:
    from src.client.simulation import SimulationSummary


@dataclass(frozen=True)
class PhaseRunProfile:
    """Runnable FL simulation profile for one dataset phase."""

    profile_id: str
    dataset_id: str
    description: str
    rounds: int
    num_clients: int
    train_mode: str
    requested_device: str
    model_id: str
    image_size: int
    train_examples_per_client: int
    eval_examples: int
    expected_per_device_train_batch_size: int
    repeat_runs: int
    repeat_tolerance: float


@dataclass(frozen=True)
class ProfileRunSummary:
    """One executed run under a phase profile."""

    run_index: int
    summary_path: str
    aggregate_eval_loss_by_round: list[float]
    final_aggregate_eval_loss: float


@dataclass(frozen=True)
class PhaseProfileReport:
    """Reference metrics and repeatability status for a phase profile run."""

    profile_id: str
    dataset_id: str
    description: str
    requested_device: str
    resolved_device: str
    used_per_device_train_batch_size: int
    expected_per_device_train_batch_size: int
    repeat_runs: int
    repeat_tolerance: float
    repeatable: bool
    baseline_rounds: int
    baseline_clients: int
    baseline_train_mode: str
    baseline_model_id: str
    baseline_image_size: int
    runs: list[ProfileRunSummary]
    max_loss_delta_vs_run_1: float


def _require_mapping(name: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    return payload


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def load_phase_run_profile(profile_path: Path) -> PhaseRunProfile:
    """Load and validate one phase run profile from YAML."""

    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    root = _require_mapping("profile", payload)
    simulation = _require_mapping("simulation", root.get("simulation"))
    expectations = _require_mapping("training_expectations", root.get("training_expectations"))
    repeatability = _require_mapping("repeatability", root.get("repeatability"))

    dataset_id = _require_str(root, "dataset_id").lower()
    if dataset_id not in MANIFEST_FILES:
        expected = ", ".join(sorted(MANIFEST_FILES))
        raise ValueError(f"dataset_id must be one of: {expected}")

    requested_device = _require_str(simulation, "requested_device")
    if requested_device not in {"cpu", "mps", "auto"}:
        raise ValueError("simulation.requested_device must be cpu, mps, or auto")

    train_mode = _require_str(simulation, "train_mode")
    if train_mode not in {"head_only", "unfreeze_last_block"}:
        raise ValueError("simulation.train_mode must be head_only or unfreeze_last_block")

    rounds = _require_int(simulation, "rounds")
    num_clients = _require_int(simulation, "num_clients")
    image_size = _require_int(simulation, "image_size")
    train_examples_per_client = _require_int(simulation, "train_examples_per_client")
    eval_examples = _require_int(simulation, "eval_examples")
    expected_batch_size = _require_int(expectations, "per_device_train_batch_size")
    repeat_runs = _require_int(repeatability, "runs")
    repeat_tolerance = _require_float(repeatability, "loss_tolerance")

    if rounds < 1:
        raise ValueError("simulation.rounds must be >= 1")
    if num_clients < 2:
        raise ValueError("simulation.num_clients must be >= 2")
    if image_size < 16:
        raise ValueError("simulation.image_size must be >= 16")
    if train_examples_per_client < 1:
        raise ValueError("simulation.train_examples_per_client must be >= 1")
    if eval_examples < 1:
        raise ValueError("simulation.eval_examples must be >= 1")
    if expected_batch_size < 1:
        raise ValueError("training_expectations.per_device_train_batch_size must be >= 1")
    if repeat_runs < 1:
        raise ValueError("repeatability.runs must be >= 1")
    if repeat_tolerance < 0:
        raise ValueError("repeatability.loss_tolerance must be >= 0")

    return PhaseRunProfile(
        profile_id=_require_str(root, "profile_id"),
        dataset_id=dataset_id,
        description=_require_str(root, "description"),
        rounds=rounds,
        num_clients=num_clients,
        train_mode=train_mode,
        requested_device=requested_device,
        model_id=_require_str(simulation, "model_id"),
        image_size=image_size,
        train_examples_per_client=train_examples_per_client,
        eval_examples=eval_examples,
        expected_per_device_train_batch_size=expected_batch_size,
        repeat_runs=repeat_runs,
        repeat_tolerance=repeat_tolerance,
    )


def _loss_series(summary: SimulationSummary) -> list[float]:
    return [float(point.aggregate_eval_loss) for point in summary.round_summaries]


def _max_abs_loss_delta(reference: list[float], candidate: list[float]) -> float:
    if len(reference) != len(candidate):
        raise ValueError(
            "Run summaries must have equal round counts for repeatability comparison"
        )
    if not reference:
        return 0.0
    return max(abs(ref - cur) for ref, cur in zip(reference, candidate, strict=True))


def run_phase_profile(
    profile: PhaseRunProfile,
    *,
    output_dir: Path,
) -> PhaseProfileReport:
    """Execute a phase run profile and return a reference report."""

    from src.client.simulation import run_local_simulation, write_simulation_summary

    resolved = resolve_run_config(
        profile.dataset_id,
        requested_device=profile.requested_device,  # type: ignore[arg-type]
        train_mode=profile.train_mode,  # type: ignore[arg-type]
        mps_available=False,
        enable_mps=profile.requested_device == "mps",
    )
    used_batch_size = int(resolved.training_args["per_device_train_batch_size"])
    if used_batch_size != profile.expected_per_device_train_batch_size:
        raise ValueError(
            "Resolved per_device_train_batch_size mismatch: "
            f"expected {profile.expected_per_device_train_batch_size}, got {used_batch_size}"
        )

    run_summaries: list[ProfileRunSummary] = []
    summaries: list[SimulationSummary] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_index in range(1, profile.repeat_runs + 1):
        run_output_dir = output_dir / f"run_{run_index:02d}"
        summary = run_local_simulation(
            dataset_id=profile.dataset_id,
            output_dir=run_output_dir,
            model_id=profile.model_id,
            requested_device=profile.requested_device,
            rounds=profile.rounds,
            num_clients=profile.num_clients,
            train_mode=profile.train_mode,
            image_size=profile.image_size,
            train_examples_per_client=profile.train_examples_per_client,
            eval_examples=profile.eval_examples,
        )
        summary_path = write_simulation_summary(
            summary,
            output_path=run_output_dir / "summary.json",
        )
        losses = _loss_series(summary)
        run_summaries.append(
            ProfileRunSummary(
                run_index=run_index,
                summary_path=str(summary_path),
                aggregate_eval_loss_by_round=losses,
                final_aggregate_eval_loss=losses[-1],
            )
        )
        summaries.append(summary)

    reference_losses = run_summaries[0].aggregate_eval_loss_by_round
    deltas = [
        _max_abs_loss_delta(reference_losses, run.aggregate_eval_loss_by_round)
        for run in run_summaries
    ]
    max_delta = max(deltas)
    repeatable = max_delta <= profile.repeat_tolerance

    return PhaseProfileReport(
        profile_id=profile.profile_id,
        dataset_id=profile.dataset_id,
        description=profile.description,
        requested_device=profile.requested_device,
        resolved_device=resolved.device,
        used_per_device_train_batch_size=used_batch_size,
        expected_per_device_train_batch_size=profile.expected_per_device_train_batch_size,
        repeat_runs=profile.repeat_runs,
        repeat_tolerance=profile.repeat_tolerance,
        repeatable=repeatable,
        baseline_rounds=summaries[0].rounds,
        baseline_clients=summaries[0].num_clients,
        baseline_train_mode=summaries[0].train_mode,
        baseline_model_id=summaries[0].model_id,
        baseline_image_size=summaries[0].image_size,
        runs=run_summaries,
        max_loss_delta_vs_run_1=max_delta,
    )


def write_phase_profile_report(report: PhaseProfileReport, *, output_path: Path) -> Path:
    """Persist a phase profile report as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
