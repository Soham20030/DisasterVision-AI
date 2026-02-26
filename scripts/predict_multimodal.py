"""
Predict damage from pre+post image pair.

Usage:
  python scripts/predict_multimodal.py --pre path/to/pre.png --post path/to/post.png
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.multimodal import CHECKPOINTS_DIR, IMG_SIZE
from multimodal.model import build_two_stream_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ["no-damage", "minor", "major", "destroyed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", required=True, help="Pre-disaster image path")
    parser.add_argument("--post", required=True, help="Post-disaster image path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model = build_two_stream_model(num_classes=4, freeze_backbone=False, dropout=0.5)
    ckpt = torch.load(CHECKPOINTS_DIR / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    pre_img = Image.open(args.pre).convert("RGB")
    post_img = Image.open(args.post).convert("RGB")
    pre_t = transform(pre_img).unsqueeze(0).to(device)
    post_t = transform(post_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pre_t, post_t)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(logits.argmax(dim=1).item())

    print(f"Prediction: {CLASS_NAMES[pred]} ({probs[pred]:.2%})")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[i]:.2%}")


if __name__ == "__main__":
    main()
