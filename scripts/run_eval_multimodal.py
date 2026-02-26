"""Evaluate multimodal model on test set."""

import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.multimodal import CHECKPOINTS_DIR, MANIFEST_PATH, NUM_CLASSES, PROCESSED_DIR
from multimodal.loaders import get_multimodal_loaders
from multimodal.model import build_two_stream_model

CLASS_NAMES = ["no-damage", "minor", "major", "destroyed"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_multimodal_loaders(MANIFEST_PATH, PROCESSED_DIR, batch_size=32, img_size=224)
    test_loader = loaders["test"]

    model = build_two_stream_model(num_classes=NUM_CLASSES, freeze_backbone=False, dropout=0.5)
    ckpt = torch.load(CHECKPOINTS_DIR / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for pre, post, labels in test_loader:
            pre, post = pre.to(device), post.to(device)
            logits = model(pre, post)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = (sum(p == l for p, l in zip(all_preds, all_labels))) / len(all_labels)
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print("Test accuracy:", f"{acc:.4f}")
    print("Test macro F1:", f"{macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures_multimodal"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Multimodal)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    print("Confusion matrix saved to", FIGURES_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
