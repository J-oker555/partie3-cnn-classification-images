from __future__ import annotations

import argparse
from pathlib import Path

from cnn_image_classification.data import describe_first_batch, make_normalized_binary_datasets


def parse_img_size(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        height, width = value.lower().split("x", 1)
        return int(height), int(width)
    size = int(value)
    return size, size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TensorFlow preprocessing pipeline for the binary image dataset.")
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--img-size", default="128", type=parse_img_size, help="Image size, for example 128 or 160x160.")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds, val_ds = make_normalized_binary_datasets(
        data_root=args.data_root,
        image_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    for name, ds in (("train", train_ds), ("val", val_ds)):
        batch = describe_first_batch(ds)
        print(f"{name} images shape: {batch['images_shape']}")
        print(f"{name} labels shape: {batch['labels_shape']}")
        print(f"{name} image values: min={batch['images_min']:.4f}, max={batch['images_max']:.4f}")
        print(f"{name} label values: min={batch['labels_min']:.0f}, max={batch['labels_max']:.0f}")


if __name__ == "__main__":
    main()
