from __future__ import annotations

import argparse
from pathlib import Path

from cnn_image_classification.data import make_mobilenet_binary_datasets
from cnn_image_classification.models import build_mobilenetv2_classifier
from cnn_image_classification.plots import plot_history
from cnn_image_classification.training import (
    best_val_accuracy,
    compile_binary_classifier,
    fit_with_timing,
    save_run_summary,
    tensorboard_callback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only the MobileNetV2 classification head.")
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--logs-dir", default=Path("logs"), type=Path)
    parser.add_argument("--outputs-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds, val_ds = make_mobilenet_binary_datasets(
        data_root=args.data_root,
        image_size=(160, 160),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    weights = None if args.weights == "none" else "imagenet"
    model, base_model = build_mobilenetv2_classifier(weights=weights)
    print(f"Nombre de couches dans base_model : {len(base_model.layers)}")
    print(f"Parametres base_model : {base_model.count_params():,}")
    compile_binary_classifier(model, learning_rate=1e-3)
    model.summary()

    history, training_time = fit_with_timing(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        callbacks=[tensorboard_callback(args.logs_dir, "transfer/head")],
    )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, "Transfer Learning tete seule", args.outputs_dir / "curves_tl_head.png")
    model.save(args.models_dir / "model_tl_head.keras")
    save_run_summary(
        args.outputs_dir / "history_tl_head.json",
        history=history,
        training_time=training_time,
        params=model.count_params(),
    )
    print(f"Temps d'entrainement tete seule : {training_time:.0f}s")
    print(f"val_accuracy max tete seule : {best_val_accuracy(history):.3f}")


if __name__ == "__main__":
    main()
