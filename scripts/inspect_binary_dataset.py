from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from cnn_image_classification.data import list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect binary image dataset counts and save a 2x3 sample grid.")
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--class-a", required=True)
    parser.add_argument("--class-b", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--output", default=Path("outputs/sample_grid.png"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_paths = {
        args.class_a: args.data_root / args.split / args.class_a,
        args.class_b: args.data_root / args.split / args.class_b,
    }

    images_by_class = {class_name: list_images(path) for class_name, path in class_paths.items()}
    for class_name, images in images_by_class.items():
        print(f"{class_paths[class_name]}: {len(images)} images")
        if len(images) < 3:
            raise SystemExit(f"Need at least 3 images for {class_name} to build the sample grid.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for row, class_name in enumerate((args.class_a, args.class_b)):
        for col, image_path in enumerate(images_by_class[class_name][:3]):
            index = row * 3 + col + 1
            image = mpimg.imread(image_path)
            print(f"{image_path}: shape={getattr(image, 'shape', 'unknown')}")
            plt.subplot(2, 3, index)
            plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
            plt.title(class_name)
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=100)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
