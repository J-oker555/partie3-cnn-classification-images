from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from cnn_image_classification.training import load_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scratch CNN and augmented CNN runs.")
    parser.add_argument("--scratch", default=Path("outputs/history_scratch.json"), type=Path)
    parser.add_argument("--augmented", default=Path("outputs/history_augmented.json"), type=Path)
    parser.add_argument("--output", default=Path("outputs/comparison_tp1_tp2.png"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch = load_run_summary(args.scratch)
    augmented = load_run_summary(args.augmented)
    scratch_history = scratch["history"]
    augmented_history = augmented["history"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(scratch_history["val_loss"], label="Scratch", color="red")
    plt.plot(augmented_history["val_loss"], label="Augmente + Dropout", color="blue")
    plt.title("Validation loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(scratch_history["val_accuracy"], label="Scratch", color="red")
    plt.plot(augmented_history["val_accuracy"], label="Augmente + Dropout", color="blue")
    plt.title("Validation accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=100)

    print("Comparaison TP1 vs TP2")
    print(f"CNN scratch           val_acc={scratch['best_val_accuracy']:.3f} params={scratch['params']:,} temps={scratch['training_time']:.0f}s")
    print(f"Augmente + Dropout    val_acc={augmented['best_val_accuracy']:.3f} params={augmented['params']:,} temps={augmented['training_time']:.0f}s")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
