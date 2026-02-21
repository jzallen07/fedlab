"""Create deterministic non-IID partitions for MedMNIST datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.data import SUPPORTED_MEDMNIST_DATASETS
from src.ml.partition import (
    SUPPORTED_SKEW_PRESETS,
    partition_medmnist_dataset,
    write_partition_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic non-IID partitions")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *SUPPORTED_MEDMNIST_DATASETS],
        help="Dataset to partition (default: all)",
    )
    parser.add_argument(
        "--root",
        default="data/medmnist",
        help="Directory where MedMNIST files are stored",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to partition",
    )
    parser.add_argument(
        "--num-clients",
        default=3,
        type=int,
        help="Number of FL clients",
    )
    parser.add_argument(
        "--seed",
        default=20260221,
        type=int,
        help="RNG seed for deterministic partitioning",
    )
    parser.add_argument(
        "--preset",
        default="moderate",
        choices=list(SUPPORTED_SKEW_PRESETS),
        help="Skew preset to apply",
    )
    parser.add_argument(
        "--image-size",
        default=224,
        type=int,
        help="Image size passed to MedMNIST dataset constructor",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing dataset files",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/partitions",
        help="Partition artifact output directory",
    )
    return parser.parse_args()


def _output_filename(dataset_id: str, split: str, preset: str, num_clients: int, seed: int) -> str:
    return f"{dataset_id}_{split}_{preset}_{num_clients}clients_seed{seed}.json"


def main() -> None:
    args = parse_args()

    dataset_ids = list(SUPPORTED_MEDMNIST_DATASETS) if args.dataset == "all" else [args.dataset]
    output_dir = Path(args.output_dir)

    for dataset_id in dataset_ids:
        result = partition_medmnist_dataset(
            dataset_id,
            root=Path(args.root),
            split=args.split,
            image_size=args.image_size,
            download=args.download,
            num_clients=args.num_clients,
            seed=args.seed,
            preset=args.preset,
        )
        output_path = write_partition_result(
            result,
            output_path=output_dir
            / _output_filename(
                dataset_id=dataset_id,
                split=args.split,
                preset=args.preset,
                num_clients=args.num_clients,
                seed=args.seed,
            ),
        )
        print(f"Wrote partition: {output_path}")


if __name__ == "__main__":
    main()
