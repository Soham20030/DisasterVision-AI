# Config for multimodal (pre+post) pipeline. Keeps single-image config unchanged.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_multimodal"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
CHECKPOINTS_DIR = PROJECT_ROOT / "outputs" / "checkpoints_multimodal"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs_multimodal"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures_multimodal"

DISASTER_TYPES = ["earthquake", "tsunami", "hurricane", "flood", "volcano", "fire"]
MAX_TOTAL_SAMPLES = 50000
MAX_SAMPLES_PER_DISASTER = 4000
MAX_NO_DAMAGE_SAMPLES = 20000

IMG_SIZE = 224
PATCH_SIZE = 256
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

NUM_CLASSES = 4
FREEZE_BACKBONE_EPOCHS = 3
DROPOUT_RATE = 0.5

BATCH_SIZE = 32
CLASS_WEIGHT_MAX = 3.0
EPOCHS = 50
LR = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 5e-3
EARLY_STOP_PATIENCE = 10
LABEL_SMOOTHING = 0.1
