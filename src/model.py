"""CNN model definitions (Transfer Learning and Custom CNN)."""
import tensorflow as tf

from src.config import IMAGE_SIZE, MODEL_ARCHITECTURE, NUM_CLASSES


def build_custom_cnn() -> tf.keras.Model:
    """Build a compact CNN from scratch; rescaling is part of the saved model."""
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
    model = tf.keras.Model(inputs, outputs, name="brain_tumor_custom_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_transfer_model() -> tf.keras.Model:
    """Build a high-accuracy Transfer Learning model using MobileNetV2 with ImageNet weights."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="mri_image")
    # MobileNetV2 preprocessing maps inputs to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Pretrained feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    # Fine-tune deeper layers while keeping early feature detectors stable
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(inputs, outputs, name="brain_tumor_mobilenet_v2")
    # Fine-tuning uses a smaller learning rate for stability
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model(architecture: str = MODEL_ARCHITECTURE) -> tf.keras.Model:
    """Build the selected model architecture ('transfer' or 'custom')."""
    if architecture == "transfer":
        return build_transfer_model()
    return build_custom_cnn()
