"""Final evaluation using only the separate Testing directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from src.config import (BATCH_SIZE, CLASS_NAMES, DISPLAY_NAMES, IMAGE_SIZE, MODEL_PATH,
                        PLOTS_DIR, RESULTS_DIR, TEST_DIR)
from src.data_preprocessing import inspect_split


def main() -> None:
    counts = inspect_split(TEST_DIR, "Testing")
    print("Evaluating only the Testing directory:", sum(counts.values()), "images")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run python src/train.py first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} model outputs; got {model.output_shape[-1]}.")
    expected_size = model.input_shape[1:3] if (model.input_shape and model.input_shape[1]) else IMAGE_SIZE
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR), labels="inferred", label_mode="int", class_names=list(CLASS_NAMES),
        image_size=expected_size, batch_size=BATCH_SIZE, shuffle=False, color_mode="rgb")
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    import numpy as np
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    print("Running batch inference on test dataset...")
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                   target_names=DISPLAY_NAMES, zero_division=0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    text = (f"Test accuracy: {accuracy:.4f}\nWeighted precision: {precision:.4f}\n"
            f"Weighted recall: {recall:.4f}\nWeighted F1-score: {f1:.4f}\n\n{report}")
    print("\n" + text)
    (RESULTS_DIR / "classification_report.txt").write_text(text, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=DISPLAY_NAMES,
                yticklabels=DISPLAY_NAMES)
    plt.xlabel("Predicted class"); plt.ylabel("True class")
    plt.title("Confusion matrix — separate test set")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
