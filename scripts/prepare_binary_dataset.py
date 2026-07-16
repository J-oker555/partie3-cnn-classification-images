from __future__ import annotations

import argparse
from pathlib import Path

from cnn_image_classification.data import list_images, split_binary_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a binary image dataset with train/val split.")
    parser.add_argument("--raw-class-a", required=True, type=Path, help="Directory containing raw images for class A.")
    parser.add_argument("--raw-class-b", required=True, type=Path, help="Directory containing raw images for class B.")
    parser.add_argument("--class-a", required=True, help="Class A folder name, for example cat.")
    parser.add_argument("--class-b", required=True, help="Class B folder name, for example dog.")
    parser.add_argument("--output-root", default=Path("data"), type=Path, help="Output dataset root.")
    parser.add_argument("--val-ratio", default=0.2, type=float, help="Validation ratio per class.")
    parser.add_argument("--seed", default=42, type=int, help="Shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files_a = list_images(args.raw_class_a)
    files_b = list_images(args.raw_class_b)

    if not files_a:
        raise SystemExit(f"No images found for class A in {args.raw_class_a}")
    if not files_b:
        raise SystemExit(f"No images found for class B in {args.raw_class_b}")

    split_binary_dataset(
        files_a=files_a,
        files_b=files_b,
        output_root=args.output_root,
        class_a=args.class_a,
        class_b=args.class_b,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for split in ("train", "val"):
        for class_name in (args.class_a, args.class_b):
            path = args.output_root / split / class_name
            print(f"{path}: {len(list(path.iterdir()))} images")


if __name__ == "__main__":
    main()
