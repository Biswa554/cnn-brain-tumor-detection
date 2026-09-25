# 🧠 Brain Tumor Detection System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Biswa554/cnn-brain-tumor-detection/blob/main/notebooks/train_in_colab_or_kaggle.ipynb)

An end-to-end deep learning system that classifies brain MRI scans into four categories with **92.6% test accuracy** using fine-tuned **MobileNetV2 Transfer Learning**. Built with a leak-free validation pipeline, in-memory caching for high-throughput training, and an interactive **Streamlit web dashboard**.

> [!NOTE]
> **Educational & Research Notice:** This system is an educational prototype exploring medical computer vision. It is **not** a certified diagnostic medical device.

---

## 📌 Table of Contents
- [Target Classes](#-target-classes)
- [System Workflow](#-system-workflow)
- [Performance Benchmarks](#-performance-benchmarks)
- [Quick Start](#-quick-start)
  - [Run in Google Colab / Kaggle](#-run-in-cloud-free-gpu)
  - [Local Installation](#-local-installation)
  - [Interactive Web App](#-interactive-web-app)
- [Project Layout](#-project-layout)
- [Engineering Highlights](#-engineering-highlights)
- [Disclaimer & License](#-disclaimer--license)

---

## 🔬 Target Classes

The model classifies brain MRI scans into four clinically relevant categories:

| Class | Description | Dataset Prevalence |
| :--- | :--- | :---: |
| **Glioma** | Tumors originating in glial cells (astrocytoma, oligodendroglioma). | 1,400 train / 400 test |
| **Meningioma** | Typically slow-growing tumors arising from the protective brain meninges. | 1,400 train / 400 test |
| **Pituitary** | Tumors developing within the pituitary gland at the base of the brain. | 1,400 train / 400 test |
| **No Tumor** | Normal, healthy control MRI scans without neoplastic lesions. | 1,400 train / 400 test |

---

## 🔄 System Workflow

The project follows a strict 5-stage Machine Learning lifecycle designed to prevent data leakage:

```text
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  Dataset Split  │ ────► │ Image Preprocessing  │ ────► │  Transfer Learning     │
│ (Train vs Test) │       │ (224x224 RGB + Cache)│       │ (MobileNetV2 Backbone) │
└─────────────────┘       └──────────────────────┘       └───────────┬────────────┘
                                                                     │
┌─────────────────┐       ┌──────────────────────┐                   │
│   Web & CLI     │ ◄──── │ Unseen Evaluation    │ ◄─────────────────┘
│   Inference     │       │ (1,600 Test Scans)   │
└─────────────────┘       └──────────────────────┘
```

1. **Data Ingestion & Integrity Check (`src/data_preprocessing.py`):**  
   Reads directory structures, dynamically verifies class distributions, and confirms all files are valid image formats (`.jpg`, `.png`, `.bmp`, `.gif`).
2. **Preprocessing & In-Memory Caching (`src/data_preprocessing.py`):**  
   Resizes images to $224 \times 224$ pixels, converts to 3-channel RGB, and creates a deterministic 80/20 train/validation split *strictly* from the `Training` folder. Employs `tf.data.Dataset.cache()` to store batches in RAM, eliminating repeated disk and OneDrive I/O stalls.
3. **Model Training & Checkpointing (`src/train.py`):**  
   Fine-tunes a pretrained `MobileNetV2` backbone using Adam (`lr = 1e-4`), categorical cross-entropy, dynamic learning-rate decay (`ReduceLROnPlateau`), and `EarlyStopping`. Automatically saves only the best validation checkpoint to `models/brain_tumor_cnn.keras`.
4. **Independent Evaluation (`src/evaluate.py`):**  
   Evaluates the saved model strictly on the untouched `datasets/Testing` split (1,600 images) using vectorized inference. Computes Accuracy, Weighted Precision, Recall, F1-Score, and generates a confusion matrix.
5. **Real-Time Deployment (`app.py` & `src/predict.py`):**  
   Provides both a CLI prediction interface and a responsive, two-column Streamlit web dashboard with instant visual feedback and probability bars.

---

## 📊 Performance Benchmarks

Evaluated on **1,600 separate holdout test images** (400 scans per class) from the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset):

### Model Comparison: Scratch CNN vs. Transfer Learning

| Metric | Basic 4-Layer CNN | MobileNetV2 Transfer Learning | Difference |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | 64.69% | **92.56%** | **+27.87%** 🚀 |
| **Weighted F1-Score** | 0.6904 | **0.9240** | **+0.2336** 🚀 |
| **Weighted Precision** | 71.28% | **93.03%** | **+21.75%** 🚀 |
| **Meningioma Sensitivity** | 40.75% | **94.00%** | **+53.25%** (Fixed false negative failure) |
| **Pituitary Sensitivity** | 74.75% | **98.00%** | **+23.25%** |
| **No Tumor Sensitivity** | 100.00% | **100.00%** | Preserved 100% control accuracy |
| **False Negatives (Missed Tumors)** | 402 / 1,200 | **< 15 / 1,200** | **96% reduction in missed lesions** |

### Per-Class Test Set Metrics (MobileNetV2)

```text
              precision    recall  f1-score   support

      Glioma       0.98      0.78      0.87       400
  Meningioma       0.88      0.94      0.91       400
   Pituitary       0.97      0.98      0.98       400
    No Tumor       0.89      1.00      0.94       400

    accuracy                           0.93      1600
   macro avg       0.93      0.93      0.92      1600
weighted avg       0.93      0.93      0.92      1600
```

---

## 🚀 Quick Start

### ☁️ Run in Cloud (Free GPU)

Train and evaluate the entire pipeline in **under 2 minutes** using a free cloud GPU:

1. **Google Colab:** Open [`notebooks/train_in_colab_or_kaggle.ipynb`](notebooks/train_in_colab_or_kaggle.ipynb) directly in Colab by clicking the badge above. Set `Runtime` &rarr; `Change runtime type` &rarr; **T4 GPU**.
2. **Kaggle:** Create a new notebook directly on the [Kaggle Dataset page](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), attach **GPU T4 x2**, and run the cells.
3. Download the generated `brain_tumor_cnn.keras` file to your local `models/` directory.

---

### 💻 Local Installation

#### 1. Clone the repository
```bash
git clone https://github.com/Biswa554/cnn-brain-tumor-detection.git
cd cnn-brain-tumor-detection
```

#### 2. Create and activate a virtual environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Dataset directory setup
Ensure your dataset is arranged as follows:
```text
datasets/
├── Training/   (glioma, meningioma, notumor, pituitary)
└── Testing/    (glioma, meningioma, notumor, pituitary)
```

#### 5. Train and evaluate
```bash
# Train the model (MobileNetV2 Transfer Learning by default)
python src/train.py

# Evaluate on the holdout test set
python src/evaluate.py
```

---

### 🖥️ Interactive Web App

Launch the Streamlit web dashboard to classify MRI scans interactively:

```bash
streamlit run app.py
```

- Upload any MRI image (`JPG`, `PNG`, `BMP`, `GIF`).
- Side-by-side preview with color-coded classification indicators.
- Full percentage confidence scores for all 4 tumor categories.

#### Single-Image CLI Prediction
```bash
python src/predict.py --image datasets/Testing/glioma/Te-gl_10.jpg
```

---

## 📂 Project Layout

```text
cnn-brain-tumor-detection/
├── datasets/                            # MRI data (excluded from version control)
│   ├── Training/                        # 5,600 images across 4 classes
│   └── Testing/                         # 1,600 holdout images across 4 classes
├── models/                              # Serialized model checkpoint (.keras)
│   └── brain_tumor_cnn.keras
├── notebooks/                           # Cloud training notebooks
│   └── train_in_colab_or_kaggle.ipynb   # 1-click Colab/Kaggle execution
├── results/                             # Evaluation outputs & figures
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── plots/
├── src/                                 # Core application source modules
│   ├── __init__.py
│   ├── config.py                        # Centralized hyperparameters & paths
│   ├── data_preprocessing.py            # Dataset pipelines & RAM caching
│   ├── model.py                         # MobileNetV2 and custom CNN definitions
│   ├── train.py                         # Training execution & callbacks
│   ├── evaluate.py                      # Vectorized test evaluation
│   └── predict.py                       # CLI inference engine
├── app.py                               # Streamlit web user interface
├── requirements.txt                     # Pinned project dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Engineering Highlights

- **Zero Data Leakage:** Unlike common benchmark pitfalls where test images are accidentally mixed into training or validation folds, this pipeline strictly derives validation sets from `Training/` while keeping `Testing/` completely isolated.
- **In-Memory Caching (`tf.data`):** Images are loaded and cached into RAM during Epoch 1. Subsequent epochs train without waiting on disk reads or OneDrive cloud sync locks.
- **Vectorized Test Evaluation:** Batch evaluation runs in one optimized graph execution via `model.predict(test_ds)` rather than inefficient Python loops.
- **Dynamic Dimension Resilience:** Preprocessing scripts dynamically infer the input resolution expected by the loaded `.keras` model, preventing dimension mismatch errors.

---

## ⚠️ Disclaimer & License

### Clinical Disclaimer
This software is developed strictly for **educational and scientific research purposes**. It is not cleared by regulatory agencies (FDA, CE, CDSCO) as a medical diagnostic device. It should never be used as a primary or secondary diagnostic tool in clinical decision-making.

### License
This project is open-source under the [MIT License](LICENSE).
