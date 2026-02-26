"""
Evaluate best model on test set: accuracy, confusion matrix, per-class metrics.

Run from project root: python scripts/run_eval.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.default import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DROPOUT_RATE,
    FIGURES_DIR,
    IMG_SIZE,
    MANIFEST_PATH,
    NUM_CLASSES,
    PROCESSED_DIR,
    ensure_dirs,
)
from src.data.loaders import get_dataloaders
from src.models.resnet_damage import build_damage_model

CLASS_NAMES = ["no-damage", "minor", "major", "destroyed"]


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(
        MANIFEST_PATH, PROCESSED_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    test_loader = loaders["test"]

    model = build_damage_model(num_classes=NUM_CLASSES, pretrained=False, dropout=DROPOUT_RATE)
    ckpt = torch.load(CHECKPOINTS_DIR / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro F1: {f1_macro:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
