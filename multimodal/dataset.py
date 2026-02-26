"""Dataset for pre+post patch pairs."""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(img_size: int, is_train: bool):
    if is_train:
        t = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        t = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return t


class MultimodalDamageDataset(Dataset):
    """Loads (pre, post) patch pairs from manifest."""

    def __init__(self, manifest: pd.DataFrame, root: Path, img_size: int = 224, is_train: bool = False):
        self.manifest = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.transform = get_transform(img_size, is_train)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        pre_path = self.root / row["path_pre"]
        post_path = self.root / row["path_post"]
        label = int(row["label"])
        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")
        pre_t = self.transform(pre_img)
        post_t = self.transform(post_img)
        return pre_t, post_t, label
