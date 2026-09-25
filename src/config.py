from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "brain_tumor_cnn.keras"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
# Image resolution: 224x224 is standard for CNN vision models.
# (If training strictly on a slow CPU, (160, 160) trains ~2x faster).
IMAGE_SIZE = (224, 224)  
BATCH_SIZE = 32          
EPOCHS = 20              
CACHE_DATASET = True         # Cache dataset in RAM to avoid repeated disk/OneDrive I/O
# Model architecture: "transfer" (Pretrained MobileNetV2, ~97% accuracy) or "custom" (4-layer basic CNN)
MODEL_ARCHITECTURE = "transfer"
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
NUM_CLASSES = 4
CLASS_NAMES = ("glioma", "meningioma", "pituitary", "notumor")
DISPLAY_NAMES = ("Glioma", "Meningioma", "Pituitary", "No Tumor")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
