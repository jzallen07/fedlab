"""Simulation smoke check for BloodMNIST phase defaults."""

from __future__ import annotations

from pathlib import Path

from src.client.simulation import run_local_simulation, write_simulation_summary


def main() -> None:
    summary = run_local_simulation(
        dataset_id="bloodmnist",
        output_dir=Path("artifacts/sim-smoke"),
        model_id="hf-internal-testing/tiny-random-DeiTForImageClassification",
        requested_device="cpu",
        rounds=1,
        num_clients=2,
        train_mode="head_only",
        image_size=30,
        train_examples_per_client=4,
        eval_examples=2,
    )
    summary_path = write_simulation_summary(
        summary,
        output_path=Path("artifacts/sim-smoke/summary.json"),
    )
    print(f"sim-smoke: ok ({summary.rounds} round, {summary.num_clients} clients)")
    print(f"sim-smoke summary: {summary_path}")


if __name__ == "__main__":
    main()
