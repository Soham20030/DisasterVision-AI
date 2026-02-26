# Default config for DisasterVision AI.
# Edit paths and hyperparameters here; no code changes needed for common tweaks.

from pathlib import Path

# ----- Paths -----
# Project root (parent of configs/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOG_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# xBD subset: all disaster types (earthquake, tsunami, hurricane, flood, volcano, fire)
DISASTER_TYPES = ["earthquake", "tsunami", "hurricane", "flood", "volcano", "fire"]
# Target total samples for training (stratified sample if we have more)
MAX_TOTAL_SAMPLES = 50000
# Max building instances per disaster type
MAX_SAMPLES_PER_DISASTER = 4000
# Cap no-damage to balance training (~40% of 50K). None = use all.
MAX_NO_DAMAGE_SAMPLES = 20000

# ----- Data -----
IMG_SIZE = 224
PATCH_SIZE = 256
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ----- Model -----
NUM_CLASSES = 4
BACKBONE = "resnet50"
# Train head only for 15 epochs, then unfreeze. Longer freeze = less overfitting.
FREEZE_BACKBONE_EPOCHS = 15
DROPOUT_RATE = 0.5

# ----- Training -----
BATCH_SIZE = 32
CLASS_WEIGHT_MAX = 3.0
EPOCHS = 50
LR = 1e-4
# Lower LR for backbone when unfrozen (avoids overwriting pretrained features)
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 5e-3
# Stop sooner when val plateaus
EARLY_STOP_PATIENCE = 5
# Label smoothing (0.1) reduces overconfident predictions
LABEL_SMOOTHING = 0.1

# ----- Logging / checkpoint -----
SAVE_EVERY_N_EPOCHS = 1
LOG_EVERY_N_STEPS = 20


def ensure_dirs():
    """Create output dirs if missing. Call once at start of train/eval scripts."""
    for d in (PROCESSED_DIR, CHECKPOINTS_DIR, LOG_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
