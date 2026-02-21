"""Run HF preprocessing for MedMNIST datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.data import SUPPORTED_MEDMNIST_DATASETS
from src.ml.preprocess import preprocess_medmnist_dataset, write_preprocess_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MedMNIST datasets for HF Trainer")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *SUPPORTED_MEDMNIST_DATASETS],
        help="Dataset to preprocess (default: all)",
    )
    parser.add_argument(
        "--root",
        default="data/medmnist",
        help="Directory where MedMNIST files are stored",
    )
    parser.add_argument(
        "--image-size",
        default=224,
        type=int,
        help="Image size for MedMNIST loader",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/deit-tiny-patch16-224",
        help="Hugging Face model id used to load AutoImageProcessor",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing dataset files",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/preprocessed",
        help="Directory where processed HF datasets are saved",
    )
    parser.add_argument(
        "--metadata-dir",
        default="artifacts/preprocess-metadata",
        help="Directory where preprocessing metadata is saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_ids = list(SUPPORTED_MEDMNIST_DATASETS) if args.dataset == "all" else [args.dataset]

    for dataset_id in dataset_ids:
        dataset_dict, metadata = preprocess_medmnist_dataset(
            dataset_id,
            root=Path(args.root),
            model_id=args.model_id,
            image_size=args.image_size,
            download=args.download,
        )

        dataset_output_path = Path(args.output_dir) / dataset_id
        metadata_output_path = Path(args.metadata_dir) / f"{dataset_id}.json"

        dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(dataset_output_path))
        write_preprocess_metadata(metadata, output_path=metadata_output_path)

        print(f"Preprocessed dataset: {dataset_id}")
        print(f"- HF dataset path: {dataset_output_path}")
        print(f"- Metadata path: {metadata_output_path}")


if __name__ == "__main__":
    main()
