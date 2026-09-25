"""Predict the four-class model output for one MRI image."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from src.config import (CLASS_NAMES, DISPLAY_NAMES, IMAGE_SIZE, MODEL_PATH,
                        SUPPORTED_EXTENSIONS)


def predict_image(image_path: str | Path, model: tf.keras.Model | None = None) -> tuple[str, float, dict[str, float]]:
    """Load, RGB-convert, resize, and classify one supported image."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format '{path.suffix}'. Use JPG, PNG, BMP, or GIF.")
    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run python src/train.py first.")
        model = tf.keras.models.load_model(MODEL_PATH)
    expected_size = model.input_shape[1:3] if (model.input_shape and model.input_shape[1]) else IMAGE_SIZE
    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(expected_size)
            array = np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read a valid image from {path}: {error}") from error
    probabilities = model.predict(array[None, ...], verbose=0)[0]
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(f"Model returned {len(probabilities)} classes; expected {len(CLASS_NAMES)}.")
    index = int(np.argmax(probabilities))
    return DISPLAY_NAMES[index], float(probabilities[index]), {
        display: float(probabilities[i]) for i, display in enumerate(DISPLAY_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one brain MRI image (research use only).")
    parser.add_argument("--image", required=True, help="Path to an MRI image (JPG/PNG/BMP/GIF).")
    args = parser.parse_args()
    try:
        label, confidence, probabilities = predict_image(args.image)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Predicted class: {label}\nConfidence: {confidence:.2%}\n")
    for name, probability in probabilities.items():
        print(f"{name}: {probability:.2%}")
    print("\nEducational and research use only; not a medical diagnosis.")


if __name__ == "__main__":
    main()
