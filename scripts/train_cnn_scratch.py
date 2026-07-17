from __future__ import annotations

import argparse
from pathlib import Path

from cnn_image_classification.data import make_normalized_binary_datasets
from cnn_image_classification.models import build_cnn_scratch
from cnn_image_classification.plots import plot_history
from cnn_image_classification.training import (
    best_val_accuracy,
    compile_binary_classifier,
    early_stopping_loss,
    first_val_loss_divergence_epoch,
    fit_with_timing,
    save_run_summary,
    tensorboard_callback,
)


def parse_img_size(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        height, width = value.lower().split("x", 1)
        return int(height), int(width)
    size = int(value)
    return size, size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the scratch CNN and save diagnostics.")
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--img-size", default="128", type=parse_img_size)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--logs-dir", default=Path("logs"), type=Path)
    parser.add_argument("--outputs-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds, val_ds = make_normalized_binary_datasets(
        data_root=args.data_root,
        image_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = build_cnn_scratch(input_shape=(args.img_size[0], args.img_size[1], 3))
    compile_binary_classifier(model)
    model.summary()

    history, training_time = fit_with_timing(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        callbacks=[
            tensorboard_callback(args.logs_dir, "scratch"),
            early_stopping_loss(),
        ],
    )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, "CNN scratch", args.outputs_dir / "curves_cnn_scratch.png")
    model.save(args.models_dir / "model_scratch.keras")
    save_run_summary(
        args.outputs_dir / "history_scratch.json",
        history=history,
        training_time=training_time,
        params=model.count_params(),
    )

    divergence_epoch = first_val_loss_divergence_epoch(history)
    print(f"Temps d'entrainement : {training_time:.0f}s")
    print(f"val_accuracy max : {best_val_accuracy(history):.3f}")
    if divergence_epoch is None:
        print("Divergence val_loss : non detectee")
    else:
        print(f"Divergence val_loss : epoch {divergence_epoch}")


if __name__ == "__main__":
    main()
