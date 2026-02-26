"""Data loaders for multimodal pipeline."""

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from multimodal.dataset import MultimodalDamageDataset


def get_multimodal_loaders(manifest_path: Path, processed_dir: Path, batch_size: int = 32, img_size: int = 224, num_workers: int = 0):
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    train_ds = MultimodalDamageDataset(train_df, processed_dir, img_size=img_size, is_train=True)
    val_ds = MultimodalDamageDataset(val_df, processed_dir, img_size=img_size, is_train=False)
    test_ds = MultimodalDamageDataset(test_df, processed_dir, img_size=img_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return {"train": train_loader, "val": val_loader, "test": test_loader}
