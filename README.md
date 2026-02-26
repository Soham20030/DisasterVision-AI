# 🌏 DisasterVision AI

**Multimodal Disaster Damage Assessment System** using satellite imagery and computer vision.

DisasterVision AI is a research-grade pipeline that analyzes pre- and post-disaster satellite images to classify building damage severity into four categories: No Damage, Minor Damage, Major Damage, and Destroyed.

## 🚀 Key Features
- **Multimodal Fusion**: Two-stream ResNet-50 architecture that compares "before" and "after" images.
- **Explainable AI**: Integrated **Grad-CAM** visualizations to highlight regions influencing model predictions.
- **High Performance**: Reaches **~85% validation accuracy** (outperforming single-image baselines).
- **Modular Pipeline**: Standardized scripts for data preparation, training, evaluation, and inference.

---

## 🛠️ Setup & Installation

### 1. Requirements
Ensure you have an NVIDIA GPU for training (CUDA 12.1 recommended).

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies (GPU-accelerated)
pip install -r requirements.txt
```

### 2. Dataset Preparation
This project uses a subset of the [xBD dataset](https://xview2.org/).
1. Place raw xBD images in `data/raw/`.
2. Run the preparation script to extract building patches:
```bash
python -m src.data.prepare_subset
```

---

## 📈 Usage

### Training
To train the multimodal model from scratch:
```bash
python scripts/run_train_multimodal.py
```

### Evaluation
To generate the classification report and confusion matrix on the test set:
```bash
python scripts/run_eval_multimodal.py
```

### Interpretability (Grad-CAM)
To visualize what the model is "looking at" for specific test samples:
```bash
python scripts/run_gradcam.py
```

### Custom Prediction
To predict damage for a specific pre/post image pair:
```bash
python scripts/predict_multimodal.py --pre path/to/pre.png --post path/to/post.png
```