from __future__ import annotations

import argparse

from cnn_image_classification.models import build_cnn_scratch, scratch_dense_params, scratch_shape_plan


def parse_img_size(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        height, width = value.lower().split("x", 1)
        return int(height), int(width)
    size = int(value)
    return size, size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and inspect the scratch CNN architecture.")
    parser.add_argument("--img-size", default="128", type=parse_img_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_shape = (args.img_size[0], args.img_size[1], 3)
    model = build_cnn_scratch(input_shape)

    print("Expected shape plan:")
    for layer_name, shape in scratch_shape_plan(args.img_size):
        print(f"- {layer_name}: {shape}")

    print(f"Dense(128) expected params: {scratch_dense_params(args.img_size):,}")
    print(f"Total params: {model.count_params():,}")
    model.summary()


if __name__ == "__main__":
    main()
