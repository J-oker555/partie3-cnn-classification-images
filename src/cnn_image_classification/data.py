from __future__ import annotations

from pathlib import Path
import random
import shutil

import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def split_binary_dataset(
    files_a: list[Path],
    files_b: list[Path],
    output_root: str | Path,
    class_a: str,
    class_b: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Copy two image lists into data/train|val/class directories."""
    output_root = Path(output_root)
    rng = random.Random(seed)

    for class_name, files in ((class_a, files_a), (class_b, files_b)):
        shuffled = list(files)
        rng.shuffle(shuffled)
        val_count = int(len(shuffled) * val_ratio)
        splits = {
            "val": shuffled[:val_count],
            "train": shuffled[val_count:],
        }
        for split, split_files in splits.items():
            target_dir = output_root / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source in split_files:
                shutil.copy2(source, target_dir / source.name)


def make_dataset(
    root: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        root,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )


def normalize_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    return ds.map(lambda images, labels: (normalization_layer(images), labels))


def optimize_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def preprocess_mobilenet_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    return ds.map(lambda images, labels: (preprocess_input(images), labels)).prefetch(tf.data.AUTOTUNE)
