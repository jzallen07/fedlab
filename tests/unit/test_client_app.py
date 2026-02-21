import argparse
import json

import pytest

from src.client import app


class _FakeNumPyClient:
    def get_parameters(self, config):
        del config
        return [1, 2, 3]

    def fit(self, parameters, config):
        del parameters, config
        return [1, 2, 3], 8, {"train_runtime": 0.1, "eval_loss": 0.7}

    def evaluate(self, parameters, config):
        del parameters, config
        return 0.6, 4, {"eval_loss": 0.6, "eval_accuracy": 0.5}


def test_parse_args_train_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["client-app", "train-step", "--client-id", "client_x", "--num-labels", "5"],
    )

    args = app.parse_args()

    assert args.command == "train-step"
    assert args.client_id == "client_x"
    assert args.num_labels == 5


def test_parse_args_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["client-app", "start", "--server-address", "127.0.0.1:8080", "--client-id", "client_1"],
    )

    args = app.parse_args()

    assert args.command == "start"
    assert args.server_address == "127.0.0.1:8080"
    assert args.client_id == "client_1"


def test_run_train_step_emits_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("src.client.app.build_numpy_client", lambda **_: _FakeNumPyClient())

    args = argparse.Namespace(
        command="train-step",
        client_id="client_0",
        dataset_id="bloodmnist",
        num_labels=3,
        train_examples=8,
        eval_examples=4,
        image_size=16,
        output_dir="/tmp/out",
        train_mode="head_only",
        device="cpu",
        model_id="fake/model",
        run_id="local-run",
        monitor_url=None,
        monitor_timeout_s=2.0,
    )

    exit_code = app.run_train_step(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["client_id"] == "client_0"
    assert payload["train_examples"] == 8
    assert payload["eval_examples"] == 4
    assert payload["loss"] == 0.6


def test_run_start_invokes_flower_client(monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, object] = {}

    monkeypatch.setattr("src.client.app.build_numpy_client", lambda **_: _FakeNumPyClient())
    monkeypatch.setattr(
        "src.client.app.fl.client.start_numpy_client",
        lambda *, server_address, client: started.update(
            {"server_address": server_address, "client": client}
        ),
    )

    args = argparse.Namespace(
        command="start",
        server_address="127.0.0.1:8080",
        client_id="client_0",
        dataset_id="bloodmnist",
        num_labels=3,
        train_examples=8,
        eval_examples=4,
        image_size=16,
        output_dir="/tmp/out",
        train_mode="head_only",
        device="cpu",
        model_id="fake/model",
        run_id="local-run",
        monitor_url=None,
        monitor_timeout_s=2.0,
    )

    exit_code = app.run_start(args)

    assert exit_code == 0
    assert started["server_address"] == "127.0.0.1:8080"
