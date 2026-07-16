from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_scratch(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_scratch",
    )


def build_data_augmentation(rotation: float = 0.1, zoom: float = 0.1) -> tf.keras.Model:
    return models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(rotation),
            layers.RandomZoom(zoom),
        ],
        name="data_augmentation",
    )


def build_cnn_augmented(
    input_shape: tuple[int, int, int],
    data_augmentation: tf.keras.Model | None = None,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    if data_augmentation is None:
        data_augmentation = build_data_augmentation()

    return models.Sequential(
        [
            layers.Input(shape=input_shape),
            data_augmentation,
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_augmented_dropout",
    )


def build_mobilenetv2_classifier(input_shape: tuple[int, int, int] = (160, 160, 3)) -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    return model, base_model


def freeze_first_layers(base_model: tf.keras.Model, ratio: float = 0.8) -> int:
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * ratio)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return fine_tune_at
