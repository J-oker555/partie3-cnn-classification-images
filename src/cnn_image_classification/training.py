from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import time

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


def fit_with_timing(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    callbacks: list[tf.keras.callbacks.Callback] | None = None,
) -> tuple[tf.keras.callbacks.History, float]:
    start = time.time()
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks or [],
    )
    return history, time.time() - start


def best_val_accuracy(history: tf.keras.callbacks.History) -> float:
    return max(history.history.get("val_accuracy", [0.0]))


def first_val_loss_divergence_epoch(history: tf.keras.callbacks.History) -> int | None:
    val_loss = history.history.get("val_loss", [])
    if len(val_loss) < 2:
        return None
    best_so_far = val_loss[0]
    for index, value in enumerate(val_loss[1:], start=2):
        if value > best_so_far:
            return index
        best_so_far = min(best_so_far, value)
    return None


def save_run_summary(
    path: str | Path,
    history: tf.keras.callbacks.History,
    training_time: float,
    params: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "training_time": float(training_time),
        "params": int(params),
        "best_val_accuracy": float(best_val_accuracy(history)),
        "divergence_epoch": first_val_loss_divergence_epoch(history),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_run_summary(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
