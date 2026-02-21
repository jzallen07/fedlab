"""Phase-aware federated simulation runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.client.simulation import run_local_simulation, write_simulation_summary
from src.ml.config import MANIFEST_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local federated simulation")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(MANIFEST_FILES.keys()),
        help="Dataset phase to simulate",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override rounds from phase manifest",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Override number of clients from phase manifest",
    )
    parser.add_argument(
        "--train-mode",
        default=None,
        choices=["head_only", "unfreeze_last_block"],
        help="Override train mode from phase manifest",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "auto"],
        help="Requested device policy",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/deit-tiny-patch16-224",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Synthetic sample image size",
    )
    parser.add_argument(
        "--train-examples-per-client",
        type=int,
        default=16,
        help="Synthetic train examples per client",
    )
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=8,
        help="Synthetic eval examples per client",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/simulation",
        help="Output directory for round artifacts",
    )
    parser.add_argument(
        "--summary-path",
        default="artifacts/simulation/summary.json",
        help="Summary JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = run_local_simulation(
        dataset_id=args.dataset,
        output_dir=Path(args.output_dir),
        model_id=args.model_id,
        requested_device=args.device,
        rounds=args.rounds,
        num_clients=args.num_clients,
        train_mode=args.train_mode,
        image_size=args.image_size,
        train_examples_per_client=args.train_examples_per_client,
        eval_examples=args.eval_examples,
    )

    summary_path = write_simulation_summary(summary, output_path=Path(args.summary_path))
    print(
        f"Completed simulation: dataset={summary.dataset_id} rounds={summary.rounds} "
        f"clients={summary.num_clients}"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
