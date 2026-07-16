from __future__ import annotations

from pathlib import Path
import os

import tensorflow as tf


def model_size_mb(path: str | Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def export_tflite(model: tf.keras.Model, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return output_path


def predict_with_tflite(model_path: str | Path, image_batch):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], image_batch)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])
