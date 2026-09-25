"""Train the CNN using Training only, with validation split from that folder."""
import sys
from pathlib import Path

# Running ``python src/train.py`` places src/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import (CLASS_NAMES, EPOCHS, MODEL_DIR, MODEL_PATH, PLOTS_DIR, RANDOM_SEED,
                        TEST_DIR, TRAIN_DIR)
from src.data_preprocessing import (inspect_split, load_training_datasets,
                                    print_dataset_analysis, sample_image_info)
from src.model import build_model


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    training_counts = inspect_split(TRAIN_DIR, "Training")
    testing_counts = inspect_split(TEST_DIR, "Testing")
    print("Class order:", ", ".join(CLASS_NAMES))
    print_dataset_analysis(training_counts, testing_counts)
    print("Sample image:", sample_image_info(TRAIN_DIR))
    print("Validation data is split only from Training; Testing remains untouched.")
    train_ds, val_ds = load_training_datasets()
    model = build_model()
    model.summary()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True,
                                           verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=3, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    # Note: ModelCheckpoint already saved the best performing model weights to MODEL_PATH.
    # Calling model.save(MODEL_PATH) here would mistakenly overwrite the best checkpoint
    # with the final (potentially overfitted) epoch weights if EarlyStopping did not trigger.
    history_path = PLOTS_DIR.parent / "training_history.csv"
    import pandas as pd
    pd.DataFrame(history.history).to_csv(history_path, index_label="epoch")
    for metric, title, filename in (("accuracy", "Accuracy", "training_accuracy.png"),
                                    ("loss", "Loss", "training_loss.png")):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label="Training")
        plt.plot(history.history[f"val_{metric}"], label="Validation")
        plt.title(f"Training and validation {title.lower()}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close()
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
