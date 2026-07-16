from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history, title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="Train loss")
    ax1.plot(history.history["val_loss"], label="Val loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="Train accuracy")
    ax2.plot(history.history["val_accuracy"], label="Val accuracy")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.show()


def plot_tp1_tp2_comparison(history_scratch, history_augmented, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_scratch.history["val_loss"], label="Scratch", color="red")
    plt.plot(history_augmented.history["val_loss"], label="Augmente + Dropout", color="blue")
    plt.title("Validation loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_scratch.history["val_accuracy"], label="Scratch", color="red")
    plt.plot(history_augmented.history["val_accuracy"], label="Augmente + Dropout", color="blue")
    plt.title("Validation accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.show()


def plot_all_models_comparison(history_scratch, history_augmented, history_tl_head, history_tl_finetune, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history_scratch.history["val_accuracy"], label="CNN scratch", color="red", linestyle="--")
    plt.plot(history_augmented.history["val_accuracy"], label="Augmente + Dropout", color="orange")
    tl_acc = history_tl_head.history["val_accuracy"] + history_tl_finetune.history["val_accuracy"]
    plt.plot(range(len(tl_acc)), tl_acc, label="MobileNetV2 fine-tuning", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.show()
