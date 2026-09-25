"""CNN model definition."""
import tensorflow as tf

from src.config import IMAGE_SIZE, NUM_CLASSES


def build_model() -> tf.keras.Model:
    """Build a compact CNN; rescaling is part of the saved model."""
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.08),
    ], name="training_augmentation")
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="mri_image")
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = augmentation(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="brain_tumor_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
