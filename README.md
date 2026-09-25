# CNN-Based Brain Tumor Detection System

An educational and research image-classification system that analyzes brain MRI scans and classifies them into four categories: **Glioma, Meningioma, Pituitary Tumor, or No Tumor**.

The system implements a leak-free Machine Learning workflow featuring **Transfer Learning (MobileNetV2)** achieving **92.6% test accuracy**, a modular Python pipeline, reproducible evaluations, and an interactive **Streamlit** web application.

> [!WARNING]
> **Educational & Research Disclaimer:** This project is an academic prototype for educational image classification and is **not intended for clinical or medical diagnostic use**. The displayed confidence represents the model's pattern score on this specific dataset, not clinical certainty.

---

## Key Features

- **Four-Class Detection:** Predicts `Glioma`, `Meningioma`, `Pituitary`, or `No Tumor`.
- **High-Accuracy Transfer Learning:** Uses a pretrained `MobileNetV2` backbone fine-tuned on brain MRI scans, reaching **92.56% accuracy** and **0.924 weighted F1-score** on unseen test data.
- **Strict Data Hygiene (Zero Leakage):** The `datasets/Testing` set is strictly isolated for final evaluation. The validation set is drawn exclusively from `datasets/Training` (80/20 split) using fixed random seeds.
- **High-Performance Pipeline:** Features in-memory caching (`tf.data.Dataset.cache()`) and parallel prefetching (`AUTOTUNE`) to eliminate disk and OneDrive I/O bottlenecks.
- **Multi-Environment Ready:** Run locally on CPU/GPU or on free cloud GPUs (**Google Colab** / **Kaggle**) in under 2 minutes.
- **Interactive Web Interface:** Streamlit app with side-by-side scan inspection, class probability bars, and model diagnostics.

---

## 5-Step Workflow

```text
MRI Datasets ──► Preprocessing & RAM Cache ──► CNN Training (MobileNetV2) ──► Unseen Test Evaluation ──► Predictions (CLI & Web)
(Train & Test)      (224x224 RGB, 80/20 Split)      (Early Stopping & Checkpoints)       (Precision/Recall/F1/Confusion)    (Streamlit & predict.py)
```

1. **Load and Inspect the Data:**  
   `src/data_preprocessing.py` checks `datasets/Training` and `datasets/Testing`, counts images across all four classes, verifies image integrity, and ensures no missing or unexpected folders.
2. **Prepare Training and Validation Data:**  
   Images are resized to $224 \times 224$ pixels and converted to 3-channel RGB. `Training` is split into an 80% training subset and a 20% validation subset. Data is cached in RAM to avoid repeated disk reads.
3. **Train the Model:**  
   `src/train.py` fine-tunes the CNN using the training split. Validation loss guides learning-rate reduction (`ReduceLROnPlateau`) and early stopping. The best model checkpoint is saved to `models/brain_tumor_cnn.keras`.
4. **Evaluate on Unseen Test Split:**  
   `src/evaluate.py` evaluates the saved model strictly on the untouched `datasets/Testing` split (1,600 images). It generates classification metrics (Accuracy, Precision, Recall, F1-score) and saves the confusion matrix heatmap.
5. **Make Predictions:**  
   Classify individual scans from the command line (`src/predict.py`) or upload scans to the Streamlit web dashboard (`app.py`) for real-time probabilities.

---

## Experimental Results

Evaluated on **1,600 separate holdout test images** (400 per class) from the Kaggle Brain Tumor MRI Dataset:

### Performance Comparison

| Metric / Class | Scratch 4-Layer CNN | MobileNetV2 (Transfer Learning) | Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | 64.69% | **92.56%** | **+27.87%** 🚀 |
| **Weighted Precision** | 71.28% | **93.03%** | **+21.75%** 🚀 |
| **Weighted Recall** | 64.69% | **92.56%** | **+27.87%** 🚀 |
| **Weighted F1-Score** | 0.6904 | **0.9240** | **+0.2336** 🚀 |
| **Meningioma Sensitivity** | 40.75% | **94.00%** | **+53.25%** (Resolved major confusion) |
| **Pituitary Sensitivity** | 74.75% | **98.00%** | Near-perfect detection |
| **No Tumor Sensitivity** | 100.00% | **100.00%** | 100% true-negative retention |
| **No Tumor Precision** | 49.88% (coin toss) | **89.00%** | False negatives dropped to minimal levels |
| **Glioma Precision** | 99.40% | **98.00%** | Virtually zero false alarms |

### Final Classification Report (MobileNetV2)

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

## Project Structure

```text
cnn-brain-tumor-detection/
├── datasets/                            # Local dataset (excluded from git)
│   ├── Training/                        # 5,600 training images (1,400 per class)
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/                         # 1,600 holdout test images (400 per class)
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── models/                              # Saved model weights (.keras)
│   └── brain_tumor_cnn.keras
├── notebooks/                           # Cloud GPU training notebooks
│   └── train_in_colab_or_kaggle.ipynb
├── results/                             # Evaluation outputs and plots
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── plots/
│       └── class_distribution.png
├── src/
│   ├── __init__.py
│   ├── config.py                        # Central settings, paths, hyperparameters
│   ├── data_preprocessing.py            # Dataset loader, validation split, RAM caching
│   ├── model.py                         # MobileNetV2 transfer learning & custom CNN
│   ├── train.py                         # Training loop, callbacks, best model saver
│   ├── evaluate.py                      # Vectorized batch evaluation & metrics
│   └── predict.py                       # Single-image inference module
├── app.py                               # Interactive Streamlit dashboard
├── requirements.txt                     # Pinned project dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/Biswa554/cnn-brain-tumor-detection.git
cd cnn-brain-tumor-detection
python -m venv .venv
```

Activate the environment:
- **Windows PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Setup

Download the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract it into the `datasets/` folder:

```text
datasets/
├── Training/ (glioma, meningioma, pituitary, notumor)
└── Testing/  (glioma, meningioma, pituitary, notumor)
```

---

## Training the Model

### Option A: Free Cloud GPU Training (Recommended — ~1.5 Minutes)
Train on an NVIDIA T4 GPU for free using the included notebook:
- **Notebook File:** [`notebooks/train_in_colab_or_kaggle.ipynb`](notebooks/train_in_colab_or_kaggle.ipynb)
- **On Google Colab:** Upload the notebook, set `Runtime` &rarr; `Change runtime type` &rarr; `T4 GPU`, and run all cells.
- **On Kaggle:** Create a new notebook directly on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), set `Accelerator` to `GPU T4 x2`, and run the notebook.
- Download `brain_tumor_cnn.keras` and place it in your local `models/` directory.

### Option B: Local Training (CPU or Local GPU)
```powershell
python src/train.py
```
*Architecture choice (`MODEL_ARCHITECTURE = "transfer"` or `"custom"`) can be configured in [`src/config.py`](src/config.py).*

---

## Evaluation

Run evaluation on the untouched `datasets/Testing` directory:

```powershell
python src/evaluate.py
```

Outputs:
- Metric summary in terminal (Accuracy, Precision, Recall, F1)
- Report file: `results/classification_report.txt`
- Heatmap: `results/confusion_matrix.png`

---

## Running Predictions

### CLI Prediction (Single Image)
```powershell
python src/predict.py --image datasets/Testing/glioma/Te-gl_10.jpg
```

Example output:
```text
Predicted class: Glioma
Confidence: 98.42%

Glioma: 98.42%
Meningioma: 1.45%
Pituitary: 0.08%
No Tumor: 0.05%

Educational and research use only; not a medical diagnosis.
```

### Streamlit Web Dashboard
Launch the web interface:

```powershell
streamlit run app.py
```

Features:
- Upload any JPG, PNG, BMP, or GIF MRI scan.
- Instant inference with cached model weights.
- Visual side-by-side scan view and probability distribution bars.

---

## Limitations & Ethical Considerations

- **Not Clinically Validated:** This model is strictly an educational demonstration. It must never be used for medical decisions, diagnosis, or treatment planning.
- **Slice vs. Volume:** MRI scans in clinical practice are 3D volumetric acquisitions (T1, T1-Gd, T2, FLAIR). This 2D slice classifier evaluates individual images without volumetric context.
- **Distributional Shift:** Performance may degrade on scans from different scanners, protocols, or patient demographics.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
