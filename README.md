# CNN-Based Brain Tumor Detection System

An academic and research project that trains a TensorFlow/Keras CNN to classify brain MRI images into Glioma, Meningioma, Pituitary, or No Tumor. It includes a reproducible training pipeline, a separate final test evaluation, single-image prediction, and a Streamlit interface.

> **Disclaimer:** This application is for educational and research purposes only and is not intended for medical diagnosis. It is not clinically validated.

## Features

- Four-class image classification using a CNN built with TensorFlow/Keras.
- Deterministic 80/20 training/validation split from `datasets/Training`.
- Separate `datasets/Testing` data used only for final evaluation.
- Training-only augmentation, early stopping, model checkpointing, and learning-rate reduction.
- Dataset class counts, sample image details, training plots, classification report, and confusion matrix.
- CLI prediction and an easy-to-use Streamlit interface.
- Dataset, trained model, and generated results are excluded from Git.

## Project architecture

```text
cnn-brain-tumor-detection/
├── datasets/                  # Local only; never commit MRI data
│   ├── Training/<class>/
│   └── Testing/<class>/
├── models/                    # Generated .keras model (ignored by Git)
├── results/                   # Generated reports and plots (ignored by Git)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset

The expected dataset contains 5,600 training images and 1,600 testing images across four classes. These counts are examples, not assumptions in the code; actual counts are detected dynamically. The canonical folder names and label order are `glioma`, `meningioma`, `pituitary`, and `notumor`.

Put the folders at the repository root as shown below (the existing repository dataset folder is `datasets`, plural):

```text
datasets/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

Images in common JPG, PNG, BMP, and GIF formats are supported. Keep all testing images out of training and validation. The loader creates its validation subset only from `Training`; no test images are used for model selection. Check for near-duplicate images across splits yourself: duplicates can inflate evaluation metrics. Patient-level splitting is preferable when patient identifiers are available.

## Technologies

Python, TensorFlow/Keras, OpenCV (available for future preprocessing extensions), NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Pillow, and Streamlit.

## Installation

Use Python 3.11 or 3.12 for the pinned TensorFlow release and a virtual environment.

```bash
git clone https://github.com/Biswa554/cnn-brain-tumor-detection.git
cd cnn-brain-tumor-detection
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation availability depends on the operating system and hardware. Use a supported Python version and consult TensorFlow's installation guidance if pip cannot install it. A GPU is optional; CPU training works but may be slower.
## Training

### Option A: Local Training (CPU or GPU)

After placing the dataset in the folders above, run from the repository root:

```bash
python src/train.py
```

Training checks both splits, prints class counts and a sample image's dimensions, creates an 80/20 split from Training, and trains for up to 20 epochs. The best validation-loss checkpoint is saved at `models/brain_tumor_cnn.keras`. Curves and CSV history are saved under `results/` and `results/plots/`.

### Option B: Free GPU Training on Google Colab or Kaggle

To train in under 2 minutes on a free NVIDIA T4 GPU:
- Use the included notebook: [`notebooks/train_in_colab_or_kaggle.ipynb`](notebooks/train_in_colab_or_kaggle.ipynb).
- **In Google Colab**: Upload the notebook, set `Runtime` -> `Change runtime type` -> `T4 GPU`, and execute the cells.
- **In Kaggle**: Create a new notebook with the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), set `Accelerator` to `GPU T4 x2`, and run the notebook.
- Download the generated `brain_tumor_cnn.keras` model to your local `models/` directory to run the Streamlit app.

## Evaluation

After training, run:

```bash
python src/evaluate.py
```

This reports accuracy and weighted precision, recall, and F1-score, with class-wise metrics and a confusion matrix. It reads only `datasets/Testing`; it must remain a final holdout and should not be used to tune choices such as the epoch or architecture. Outputs: `results/classification_report.txt`, `results/confusion_matrix.png`, and `results/plots/confusion_matrix.png`.

## Single-image prediction

```bash
python src/predict.py --image path/to/image.jpg
```

The script prints the predicted class, confidence score, and all four class probabilities. Prediction requires the trained model file.

## Streamlit application

```bash
streamlit run app.py
```

Upload a JPG, PNG, BMP, or GIF MRI image and choose **Predict class**. The model loads once and is cached during the Streamlit session.

## Model architecture

The model converts pixels to the `[0, 1]` range, applies conservative random augmentation during training, then uses four convolution blocks (32, 64, 128, and 192 filters) to learn increasingly detailed image patterns. Batch normalization stabilizes intermediate activations, and max pooling reduces spatial size. Global average pooling summarizes each feature map; a 128-unit dense layer and dropout form the classifier head. The final four-unit softmax produces class probabilities. Training uses Adam and categorical cross-entropy.

## Workflow

```text
MRI dataset → resize/RGB input → Training-only train/validation split
            → CNN training → best validation checkpoint
            → final evaluation on Testing → CLI or web prediction
```

## Evaluation metrics

- **Accuracy:** fraction of all test predictions that are correct.
- **Precision:** among predictions for a class, the fraction that are correct.
- **Recall:** among examples of a class, the fraction found by the model.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts true classes against predicted classes to show error patterns.

## Results

No performance results are supplied here. Run training and the final evaluation on your local dataset, then record the measured metrics here; do not treat them as clinical performance.

## Limitations

This model is an educational prototype, not a medical diagnostic device. It has not been clinically validated and may fail on images from different scanners, protocols, populations, or preprocessing pipelines. Folder-based random splitting cannot ensure patient-level separation. Dataset quality, duplicates, class imbalance, and acquisition bias can affect the measured results. Model confidence is not medical certainty.

## Future improvements

- Compare with transfer-learning baselines.
- Use larger and more diverse datasets with patient-level split metadata.
- Add duplicate/near-duplicate detection and dataset quality checks.
- Add Grad-CAM for exploratory interpretability, without treating it as proof of correctness.
- Explore model calibration, optimization, and deployment controls.

## License

See [LICENSE](LICENSE).
