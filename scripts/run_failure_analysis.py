"""
Failure analysis: identify misclassified test samples and run Grad-CAM on them.

Saves failure cases with Grad-CAM overlays to outputs/figures/failure_analysis/.
Run from project root: python scripts/run_failure_analysis.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

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
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.data.loaders import get_dataloaders
from src.models.resnet_damage import build_damage_model
from src.visualization.gradcam import GradCAM, overlay_heatmap

CLASS_NAMES = ["no-damage", "minor", "major", "destroyed"]


def main():
    ensure_dirs()
    out_dir = FIGURES_DIR / "failure_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_damage_model(num_classes=NUM_CLASSES, pretrained=False, dropout=DROPOUT_RATE)
    ckpt = torch.load(CHECKPOINTS_DIR / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    loaders = get_dataloaders(MANIFEST_PATH, PROCESSED_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    test_loader = loaders["test"]

    # Collect all predictions and labels
    all_preds, all_labels, all_paths = [], [], []
    manifest = pd.read_csv(MANIFEST_PATH)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    with torch.no_grad():
        idx = 0
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            for i in range(images.size(0)):
                all_preds.append(preds[i].item())
                all_labels.append(labels[i].item())
                all_paths.append(test_df.iloc[idx]["path"])
                idx += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    failures = np.where(all_preds != all_labels)[0]
    n_failures = len(failures)

    print(f"Test set: {len(all_preds)} samples, {n_failures} misclassified ({100 * n_failures / len(all_preds):.1f}%)")

    # Save failure summary
    summary_path = out_dir / "failure_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total test samples: {len(all_preds)}\n")
        f.write(f"Misclassified: {n_failures} ({100 * n_failures / len(all_preds):.1f}%)\n\n")
        f.write("Failure breakdown by true class:\n")
        for c in range(NUM_CLASSES):
            mask = (all_labels[failures] == c)
            f.write(f"  {CLASS_NAMES[c]}: {mask.sum()} failures\n")
    print(f"Summary saved to {summary_path}")

    # Grad-CAM on a sample of failures (max 12)
    gradcam = GradCAM(model, model.layer4)
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    n_show = min(12, n_failures)
    indices = np.random.RandomState(42).choice(failures, n_show, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for i, fail_idx in enumerate(indices):
        if i >= len(axes):
            break
        path = PROCESSED_DIR / all_paths[fail_idx]
        label = all_labels[fail_idx]
        pred = all_preds[fail_idx]

        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.enable_grad():
            heatmap = gradcam(img_tensor)
        overlay = overlay_heatmap(img_np, heatmap, alpha=0.5)

        axes[i].imshow(overlay)
        axes[i].set_title(
            f"True: {CLASS_NAMES[label]} | Pred: {CLASS_NAMES[pred]}\n{path.name}"
        )
        axes[i].axis("off")

    plt.suptitle("Failure Analysis: Grad-CAM on misclassified samples", fontsize=12)
    plt.tight_layout()
    fig_path = out_dir / "failure_gradcam.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Grad-CAM on failures saved to {fig_path}")


if __name__ == "__main__":
    main()
