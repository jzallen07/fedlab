"""Client CLI for local smoke validation of HF Flower client lifecycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flwr as fl
import torch

from src.client.hf_client import HFClientConfig, build_numpy_client
from src.ml.config import MANIFEST_FILES, load_manifest


class _SyntheticVisionDataset(torch.utils.data.Dataset):
    def __init__(self, *, num_examples: int, num_labels: int, image_size: int = 16) -> None:
        generator = torch.Generator().manual_seed(20260221)
        self.pixel_values = torch.rand(
            num_examples,
            3,
            image_size,
            image_size,
            generator=generator,
        )
        self.labels = torch.tensor(
            [idx % num_labels for idx in range(num_examples)],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": self.pixel_values[index],
            "labels": self.labels[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF Flower client utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_step = subparsers.add_parser(
        "train-step",
        help="Run one local synthetic fit/eval cycle and print metrics",
    )
    train_step.add_argument("--client-id", default="client_0")
    train_step.add_argument("--dataset-id", default="bloodmnist")
    train_step.add_argument("--num-labels", type=int, default=3)
    train_step.add_argument("--train-examples", type=int, default=8)
    train_step.add_argument("--eval-examples", type=int, default=4)
    train_step.add_argument("--image-size", type=int, default=16)
    train_step.add_argument("--output-dir", default="artifacts/client-smoke")
    train_step.add_argument(
        "--train-mode",
        default="head_only",
        choices=["head_only", "unfreeze_last_block"],
    )
    train_step.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    train_step.add_argument("--model-id", default="facebook/deit-tiny-patch16-224")
    train_step.add_argument("--run-id", default="local-run")
    train_step.add_argument("--monitor-url", default=None)
    train_step.add_argument("--monitor-timeout-s", type=float, default=2.0)

    start = subparsers.add_parser(
        "start",
        help="Start a Flower NumPy client and connect to a running server",
    )
    start.add_argument("--server-address", required=True)
    start.add_argument("--client-id", default="client_0")
    start.add_argument(
        "--dataset-id",
        default="bloodmnist",
        choices=sorted(MANIFEST_FILES),
    )
    start.add_argument(
        "--num-labels",
        type=int,
        default=None,
        help="Override label count (defaults to dataset manifest value)",
    )
    start.add_argument("--train-examples", type=int, default=16)
    start.add_argument("--eval-examples", type=int, default=8)
    start.add_argument("--image-size", type=int, default=224)
    start.add_argument("--output-dir", default="artifacts/clients")
    start.add_argument(
        "--train-mode",
        default="head_only",
        choices=["head_only", "unfreeze_last_block"],
    )
    start.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    start.add_argument("--model-id", default="facebook/deit-tiny-patch16-224")
    start.add_argument("--run-id", default="local-run")
    start.add_argument("--monitor-url", default=None)
    start.add_argument("--monitor-timeout-s", type=float, default=2.0)

    return parser.parse_args()


def run_train_step(args: argparse.Namespace) -> int:
    train_dataset = _SyntheticVisionDataset(
        num_examples=args.train_examples,
        num_labels=args.num_labels,
        image_size=args.image_size,
    )
    eval_dataset = _SyntheticVisionDataset(
        num_examples=args.eval_examples,
        num_labels=args.num_labels,
        image_size=args.image_size,
    )

    client = build_numpy_client(
        config=HFClientConfig(
            client_id=args.client_id,
            dataset_id=args.dataset_id,
            num_labels=args.num_labels,
            output_dir=Path(args.output_dir),
            model_id=args.model_id,
            requested_device=args.device,
            default_train_mode=args.train_mode,
            run_id=args.run_id,
            monitor_url=args.monitor_url,
            monitor_timeout_s=args.monitor_timeout_s,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        label_names=[f"label_{idx}" for idx in range(args.num_labels)],
    )

    initial = client.get_parameters({})
    updated, train_count, train_metrics = client.fit(
        initial,
        {
            "train_mode": args.train_mode,
            "round": 1,
        },
    )
    loss, eval_count, eval_metrics = client.evaluate(
        updated,
        {
            "train_mode": args.train_mode,
        },
    )

    print(
        json.dumps(
            {
                "client_id": args.client_id,
                "train_examples": train_count,
                "eval_examples": eval_count,
                "loss": loss,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_num_labels(args: argparse.Namespace) -> int:
    if args.num_labels is not None:
        return int(args.num_labels)
    manifest = load_manifest(str(args.dataset_id))
    return int(manifest.num_labels)


def run_start(args: argparse.Namespace) -> int:
    num_labels = _resolve_num_labels(args)
    train_dataset = _SyntheticVisionDataset(
        num_examples=args.train_examples,
        num_labels=num_labels,
        image_size=args.image_size,
    )
    eval_dataset = _SyntheticVisionDataset(
        num_examples=args.eval_examples,
        num_labels=num_labels,
        image_size=args.image_size,
    )
    label_names = [f"label_{idx}" for idx in range(num_labels)]

    client = build_numpy_client(
        config=HFClientConfig(
            client_id=args.client_id,
            dataset_id=args.dataset_id,
            num_labels=num_labels,
            output_dir=Path(args.output_dir),
            model_id=args.model_id,
            requested_device=args.device,
            default_train_mode=args.train_mode,
            run_id=args.run_id,
            monitor_url=args.monitor_url,
            monitor_timeout_s=args.monitor_timeout_s,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        label_names=label_names,
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)
    return 0


def main() -> int:
    args = parse_args()

    if args.command == "train-step":
        return run_train_step(args)
    if args.command == "start":
        return run_start(args)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
