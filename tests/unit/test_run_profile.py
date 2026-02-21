from pathlib import Path

import pytest

from src.client.simulation import RoundSummary, SimulationSummary
from src.ml.run_profile import (
    PhaseProfileReport,
    load_phase_run_profile,
    run_phase_profile,
    write_phase_profile_report,
)


def _summary(losses: list[float]) -> SimulationSummary:
    rounds = [
        RoundSummary(
            round=index + 1,
            aggregate_eval_loss=loss,
            client_train_examples={"client_0": 4, "client_1": 4},
            client_eval_examples={"client_0": 2, "client_1": 2},
            client_train_metrics={"client_0": {}, "client_1": {}},
            client_eval_metrics={"client_0": {}, "client_1": {}},
        )
        for index, loss in enumerate(losses)
    ]
    return SimulationSummary(
        dataset_id="bloodmnist",
        phase="Phase A",
        rounds=len(rounds),
        num_clients=2,
        train_mode="head_only",
        device="cpu",
        model_id="facebook/deit-tiny-patch16-224",
        image_size=224,
        round_summaries=rounds,
    )


def test_load_phase_run_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "profile_id: bloodmnist_baseline_cpu",
                "dataset_id: bloodmnist",
                "description: baseline",
                "simulation:",
                "  rounds: 1",
                "  num_clients: 2",
                "  train_mode: head_only",
                "  requested_device: cpu",
                "  model_id: facebook/deit-tiny-patch16-224",
                "  image_size: 224",
                "  train_examples_per_client: 4",
                "  eval_examples: 2",
                "training_expectations:",
                "  per_device_train_batch_size: 8",
                "repeatability:",
                "  runs: 2",
                "  loss_tolerance: 1.0e-9",
            ]
        ),
        encoding="utf-8",
    )

    profile = load_phase_run_profile(profile_path)

    assert profile.dataset_id == "bloodmnist"
    assert profile.rounds == 1
    assert profile.repeat_runs == 2


@pytest.mark.parametrize(
    ("profile_path", "dataset_id"),
    [
        ("configs/profiles/bloodmnist_baseline.yaml", "bloodmnist"),
        ("configs/profiles/dermamnist_extension.yaml", "dermamnist"),
        ("configs/profiles/pathmnist_extension.yaml", "pathmnist"),
    ],
)
def test_repository_profile_configs_load(profile_path: str, dataset_id: str) -> None:
    profile = load_phase_run_profile(Path(profile_path))
    assert profile.dataset_id == dataset_id
    assert profile.requested_device == "cpu"
    assert profile.repeat_runs >= 1


def test_run_phase_profile_repeatable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs = [_summary([0.8]), _summary([0.8])]
    state = {"index": 0}

    def _fake_run_local_simulation(**kwargs):
        del kwargs
        index = state["index"]
        state["index"] += 1
        return runs[index]

    def _fake_write_simulation_summary(summary, *, output_path: Path):
        del summary
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return output_path

    class _Resolved:
        device = "cpu"
        training_args = {"per_device_train_batch_size": 8}

    def _fake_resolve_run_config(*args, **kwargs):
        del args, kwargs
        return _Resolved()

    monkeypatch.setattr(
        "src.client.simulation.run_local_simulation",
        _fake_run_local_simulation,
    )
    monkeypatch.setattr(
        "src.client.simulation.write_simulation_summary",
        _fake_write_simulation_summary,
    )
    monkeypatch.setattr(
        "src.ml.run_profile.resolve_run_config",
        _fake_resolve_run_config,
    )

    profile = load_phase_run_profile(
        Path("configs/profiles/bloodmnist_baseline.yaml")
    )
    report = run_phase_profile(profile, output_dir=tmp_path / "profile")

    assert report.repeatable is True
    assert report.max_loss_delta_vs_run_1 == 0.0
    assert len(report.runs) == 2


def test_run_phase_profile_detects_non_repeatable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs = [_summary([0.9]), _summary([0.5])]
    state = {"index": 0}

    def _fake_run_local_simulation(**kwargs):
        del kwargs
        index = state["index"]
        state["index"] += 1
        return runs[index]

    def _fake_write_simulation_summary(summary, *, output_path: Path):
        del summary
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return output_path

    class _Resolved:
        device = "cpu"
        training_args = {"per_device_train_batch_size": 8}

    def _fake_resolve_run_config(*args, **kwargs):
        del args, kwargs
        return _Resolved()

    monkeypatch.setattr(
        "src.client.simulation.run_local_simulation",
        _fake_run_local_simulation,
    )
    monkeypatch.setattr(
        "src.client.simulation.write_simulation_summary",
        _fake_write_simulation_summary,
    )
    monkeypatch.setattr(
        "src.ml.run_profile.resolve_run_config",
        _fake_resolve_run_config,
    )

    profile = load_phase_run_profile(
        Path("configs/profiles/bloodmnist_baseline.yaml")
    )
    report = run_phase_profile(profile, output_dir=tmp_path / "profile")

    assert report.repeatable is False
    assert report.max_loss_delta_vs_run_1 > 0.0


def test_write_phase_profile_report(tmp_path: Path) -> None:
    profile = load_phase_run_profile(
        Path("configs/profiles/bloodmnist_baseline.yaml")
    )
    report = PhaseProfileReport(
        profile_id=profile.profile_id,
        dataset_id=profile.dataset_id,
        description=profile.description,
        requested_device=profile.requested_device,
        resolved_device="cpu",
        used_per_device_train_batch_size=8,
        expected_per_device_train_batch_size=8,
        repeat_runs=2,
        repeat_tolerance=1.0e-9,
        repeatable=True,
        baseline_rounds=1,
        baseline_clients=2,
        baseline_train_mode="head_only",
        baseline_model_id="facebook/deit-tiny-patch16-224",
        baseline_image_size=224,
        runs=[],
        max_loss_delta_vs_run_1=0.0,
    )
    output_path = write_phase_profile_report(
        report,
        output_path=tmp_path / "profile" / "report.json",
    )

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "bloodmnist_baseline_cpu" in text
    assert '"repeatable": true' in text
