from __future__ import annotations

import datetime as dt
from pathlib import Path

import tensorflow as tf


def compile_binary_classifier(model: tf.keras.Model, learning_rate: float | None = None) -> None:
    optimizer = "adam" if learning_rate is None else tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])


def tensorboard_callback(logs_root: str | Path, run_name: str) -> tf.keras.callbacks.TensorBoard:
    log_dir = Path(logs_root) / run_name / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)


def early_stopping_loss() -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )


def early_stopping_accuracy() -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    )
