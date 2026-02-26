"""
Run Grad-CAM on sample test images or a specific image.

Shows which regions the model used for its prediction. Saves to outputs/figures/gradcam/.
Usage:
  python scripts/run_gradcam.py              # random test samples
  python scripts/run_gradcam.py image.png   # single image
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.default import (
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
from src.models.resnet_damage import build_damage_model
from src.visualization.gradcam import GradCAM, overlay_heatmap

CLASS_NAMES = ["no-damage", "minor", "major", "destroyed"]


def run_single_image(gradcam, model, transform, path, device, gradcam_dir):
    """Run Grad-CAM on a single image and save."""
    path = Path(path)
    if not path.is_file():
        print(f"File not found: {path}")
        return
    img_pil = Image.open(path).convert("RGB")
    img_np = np.array(img_pil)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.enable_grad():
        heatmap = gradcam(img_tensor)
    overlay = overlay_heatmap(img_np, heatmap, alpha=0.5)

    with torch.no_grad():
        pred_logits = model(img_tensor)
    pred_class = pred_logits.argmax(dim=1).item()
    pred_prob = torch.softmax(pred_logits, dim=1)[0, pred_class].item()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(overlay)
    ax.set_title(f"Pred: {CLASS_NAMES[pred_class]} ({pred_prob:.2%})\n{path.name}")
    ax.axis("off")
    out_path = gradcam_dir / f"gradcam_{path.stem}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grad-CAM saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM on images")
    parser.add_argument("image", nargs="?", help="Optional: path to single image (e.g. wreckage.png)")
    args = parser.parse_args()

    ensure_dirs()
    gradcam_dir = FIGURES_DIR / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_damage_model(num_classes=NUM_CLASSES, pretrained=False, dropout=DROPOUT_RATE)
    ckpt = torch.load(CHECKPOINTS_DIR / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer)
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if args.image:
        run_single_image(gradcam, model, transform, args.image, device, gradcam_dir)
        return

    # Default: random test samples
    import pandas as pd
    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    n_samples = min(12, len(test_df))
    indices = np.random.RandomState(42).choice(len(test_df), n_samples, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
        row = test_df.iloc[idx]
        path = PROCESSED_DIR / row["path"]
        label = int(row["label"])

        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.enable_grad():
            heatmap = gradcam(img_tensor)
        overlay = overlay_heatmap(img_np, heatmap, alpha=0.5)

        with torch.no_grad():
            pred_logits = model(img_tensor)
        pred_class = pred_logits.argmax(dim=1).item()
        pred_prob = torch.softmax(pred_logits, dim=1)[0, pred_class].item()
        correct = pred_class == label

        axes[i].imshow(overlay)
        axes[i].set_title(
            f"True: {CLASS_NAMES[label]} | Pred: {CLASS_NAMES[pred_class]} ({pred_prob:.2f})\n{'✓' if correct else '✗'}"
        )
        axes[i].axis("off")

    plt.suptitle("Grad-CAM: Regions influencing damage classification", fontsize=12)
    plt.tight_layout()
    out_path = gradcam_dir / "gradcam_samples.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grad-CAM saved to {out_path}")


if __name__ == "__main__":
    main()
