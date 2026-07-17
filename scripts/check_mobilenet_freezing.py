from __future__ import annotations

import argparse

from cnn_image_classification.models import build_mobilenetv2_classifier, count_trainable_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MobileNetV2 transfer model and verify freezing.")
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = None if args.weights == "none" else "imagenet"
    model, base_model = build_mobilenetv2_classifier(weights=weights)

    print(f"Base layers: {len(base_model.layers)}")
    print(f"Base params: {base_model.count_params():,}")
    print(f"Base trainable: {base_model.trainable}")
    print(f"Model trainable params: {count_trainable_params(model):,}")
    model.summary()

    if base_model.trainable:
        raise SystemExit("Expected MobileNetV2 base to be frozen.")
    if count_trainable_params(base_model) != 0:
        raise SystemExit("Expected zero trainable params in MobileNetV2 base.")


if __name__ == "__main__":
    main()
