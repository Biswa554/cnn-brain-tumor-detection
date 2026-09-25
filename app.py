"""Streamlit interface for the educational MRI classification model."""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import CLASS_NAMES, DISPLAY_NAMES, IMAGE_SIZE, MODEL_PATH, SUPPORTED_EXTENSIONS

st.set_page_config(page_title="Brain Tumor CNN", page_icon="🧠", layout="centered")
st.title("CNN-Based Brain Tumor Detection System")
st.write("An educational demonstration of four-class MRI image classification.")
st.warning("This application is for educational and research purposes only and is not intended for medical diagnosis.")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it with `python src/train.py` first.")
    return tf.keras.models.load_model(MODEL_PATH)


uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png", "bmp", "gif"])
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            st.image(image, caption="Uploaded MRI scan", use_container_width=True)
            predict_btn = st.button("🔍 Predict Class", type="primary", use_container_width=True)

        with col2:
            if predict_btn:
                with st.spinner("Analyzing image..."):
                    model = load_model()
                    expected_size = model.input_shape[1:3] if (model.input_shape and model.input_shape[1]) else IMAGE_SIZE
                    resized = image.resize(expected_size)
                    probabilities = model.predict(np.asarray(resized, dtype=np.float32)[None, ...], verbose=0)[0]
                
                if len(probabilities) != len(CLASS_NAMES):
                    st.error("The loaded model does not have four outputs. Please retrain the project model.")
                else:
                    index = int(np.argmax(probabilities))
                    predicted_label = DISPLAY_NAMES[index]
                    confidence = float(probabilities[index])

                    # Status alert based on class
                    if predicted_label == "No Tumor":
                        st.success(f"**Prediction: {predicted_label}**")
                    else:
                        st.info(f"**Prediction: {predicted_label}**")

                    st.metric("Model Confidence", f"{confidence:.1%}")
                    st.markdown("#### Class Probabilities")
                    for name, prob in zip(DISPLAY_NAMES, probabilities):
                        st.progress(float(prob), text=f"{name}: {prob:.1%}")
                    
                    st.caption("⚠️ *Confidence reflects the model's pattern score on this dataset, not a clinical diagnosis.*")
            else:
                st.info("Click **🔍 Predict Class** to analyze the uploaded scan.")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"The uploaded file could not be read as an image: {error}")
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not run prediction: {error}")
