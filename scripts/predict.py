"""
Inference CLI: predict damage class for image(s).

Usage:
  python scripts/predict.py path/to/image.png
  python scripts/predict.py path/to/folder/
  python scripts/predict.py img1.png img2.png --save
"""

import argparse
import sys
from pathlib import Path

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
    NUM_CLASSES,
    ensure_dirs,
)
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.resnet_damage import build_damage_model

CLASS_NAMES = ["no-damage", "minor", "major", "destroyed"]


def load_model(device):
    model = build_damage_model(num_classes=NUM_CLASSES, pretrained=False, dropout=DROPOUT_RATE)
    ckpt = torch.load(CHECKPOINTS_DIR / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


def get_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict_image(model, transform, path, device):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(logits.argmax(dim=1).item())
    return pred, probs


def main():
    parser = argparse.ArgumentParser(description="Predict damage class for satellite image(s)")
    parser.add_argument("paths", nargs="+", help="Image path(s) or folder path")
    parser.add_argument("--save", action="store_true", help="Save predictions to outputs/figures/predictions/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    transform = get_transform()

    # Resolve paths: if single path is a dir, glob images
    all_paths = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif"):
                all_paths.extend(path.glob(ext))
        elif path.is_file():
            all_paths.append(path)
        else:
            print(f"Warning: {p} not found, skipping")
    all_paths = sorted(set(all_paths))

    if not all_paths:
        print("No images found.")
        return

    if args.save:
        ensure_dirs()
        out_dir = FIGURES_DIR / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

    for path in all_paths:
        pred, probs = predict_image(model, transform, path, device)
        name = CLASS_NAMES[pred]
        conf = probs[pred]
        print(f"{path.name}: {name} ({conf:.2%})")
        for i, cname in enumerate(CLASS_NAMES):
            print(f"  {cname}: {probs[i]:.2%}")
        if args.save:
            out_path = out_dir / f"{path.stem}_pred_{name}.txt"
            with open(out_path, "w") as f:
                f.write(f"File: {path.name}\nPredicted: {name} ({conf:.2%})\n")
                for i, cname in enumerate(CLASS_NAMES):
                    f.write(f"  {cname}: {probs[i]:.2%}\n")
            print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
