from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from cnn_image_classification.data import make_dataset
from cnn_image_classification.models import build_data_augmentation


def parse_img_size(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        height, width = value.lower().split("x", 1)
        return int(height), int(width)
    size = int(value)
    return size, size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a 3x3 preview grid for Keras data augmentation.")
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--img-size", default="128", type=parse_img_size)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--rotation", default=0.1, type=float)
    parser.add_argument("--zoom", default=0.1, type=float)
    parser.add_argument("--output", default=Path("outputs/augmentation_grid.png"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = make_dataset(
        args.data_root / "train",
        image_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    images, labels = next(iter(train_ds))
    sample_image = images[0]
    sample_label = labels[0].numpy().item()

    data_augmentation = build_data_augmentation(rotation=args.rotation, zoom=args.zoom)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    for index in range(9):
        augmented = data_augmentation(tf.expand_dims(sample_image, 0), training=True)
        plt.subplot(3, 3, index + 1)
        plt.imshow(tf.cast(tf.clip_by_value(augmented[0], 0, 255), tf.uint8))
        plt.axis("off")
    plt.suptitle(f"Augmentation preview - label={sample_label:.0f}")
    plt.tight_layout()
    plt.savefig(args.output, dpi=100)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
