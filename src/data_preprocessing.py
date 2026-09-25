"""Dataset inspection and TensorFlow input pipeline helpers."""
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from src.config import (BATCH_SIZE, CACHE_DATASET, CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED,
                        SUPPORTED_EXTENSIONS, TRAIN_DIR, VALIDATION_SPLIT)


def inspect_split(directory: Path, split_name: str) -> dict[str, int]:
    """Count supported images and reject missing/empty or unexpected classes."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{split_name} directory not found: {directory}")
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_dir = directory / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder '{class_name}' in {directory}")
        count = sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if count == 0:
            raise ValueError(f"Class folder is empty (no supported images): {class_dir}")
        counts[class_name] = count
    extras = [p.name for p in directory.iterdir() if p.is_dir() and p.name.lower() not in CLASS_NAMES]
    if extras:
        raise ValueError(f"Unexpected class folders in {directory}: {', '.join(extras)}")
    return counts


def print_dataset_analysis(training: dict[str, int], testing: dict[str, int]) -> None:
    """Print split sizes and write a class distribution chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\nDataset statistics (image files):")
    for name, counts in (("Training", training), ("Testing", testing)):
        print(f"{name}: " + ", ".join(f"{c}: {counts[c]}" for c in CLASS_NAMES) + f" | total: {sum(counts.values())}")
    from src.config import PLOTS_DIR
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    values = [training[c] for c in CLASS_NAMES] + [testing[c] for c in CLASS_NAMES]
    labels = [f"Train: {c}" for c in CLASS_NAMES] + [f"Test: {c}" for c in CLASS_NAMES]
    plt.figure(figsize=(11, 5))
    sns.barplot(x=labels, y=values, hue=["Training"] * 4 + ["Testing"] * 4, legend=False)
    plt.title("Dataset class distribution")
    plt.ylabel("Number of images")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=150)
    plt.close()


def sample_image_info(directory: Path) -> str:
    """Return dimensions and format for one readable sample image."""
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                with Image.open(path) as image:
                    return f"{path.name}: {image.size[0]}x{image.size[1]} pixels, {image.format}, {image.mode}"
            except (OSError, UnidentifiedImageError):
                continue
    return "No readable sample image found."


def load_training_datasets(image_size: tuple[int, int] = IMAGE_SIZE) -> tuple[Any, Any]:
    """Create disjoint training/validation subsets from Training only."""
    common = dict(directory=str(TRAIN_DIR), labels="inferred", label_mode="categorical",
                  class_names=list(CLASS_NAMES), image_size=image_size,
                  batch_size=BATCH_SIZE, seed=RANDOM_SEED, validation_split=VALIDATION_SPLIT,
                  color_mode="rgb")
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", shuffle=False, **common)
    autotune = tf.data.AUTOTUNE
    if CACHE_DATASET:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune)
