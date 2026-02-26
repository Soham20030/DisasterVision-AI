"""
Train damage classification model (ResNet-50 + 4-class head).

Uses class weights from training set, early stopping, and checkpointing.
Run from project root: python scripts/run_train.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.default import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CLASS_WEIGHT_MAX,
    DROPOUT_RATE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    ensure_dirs,
    FIGURES_DIR,
    FREEZE_BACKBONE_EPOCHS,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LR,
    LR_BACKBONE,
    MANIFEST_PATH,
    NUM_CLASSES,
    PROCESSED_DIR,
    SAVE_EVERY_N_EPOCHS,
    LOG_EVERY_N_STEPS,
    WEIGHT_DECAY,
)
from src.data.loaders import get_dataloaders
from src.models.resnet_damage import build_damage_model, unfreeze_backbone


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Data
    loaders = get_dataloaders(
        MANIFEST_PATH, PROCESSED_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    # Class weights from training set (inverse frequency)
    import pandas as pd
    manifest = pd.read_csv(MANIFEST_PATH)
    train_labels = manifest[manifest["split"] == "train"]["label"].values
    classes = np.unique(train_labels)
    weights = compute_class_weight(
        "balanced", classes=classes, y=train_labels
    )
    # Cap weights to reduce over-prediction of rare classes (e.g. "destroyed")
    weights = np.minimum(weights, CLASS_WEIGHT_MAX)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Class weights:", class_weights.tolist())

    # Model
    model = build_damage_model(
        num_classes=NUM_CLASSES,
        freeze_backbone=(FREEZE_BACKBONE_EPOCHS > 0),
        pretrained=True,
        dropout=DROPOUT_RATE,
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_BACKBONE_EPOCHS + 1 and FREEZE_BACKBONE_EPOCHS > 0:
            unfreeze_backbone(model)
            backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
            head_params = [p for n, p in model.named_parameters() if "fc" in n]
            optimizer = torch.optim.AdamW(
                [{"params": backbone_params, "lr": LR_BACKBONE}, {"params": head_params, "lr": LR}],
                weight_decay=WEIGHT_DECAY,
            )
            print("Unfroze backbone at epoch", epoch, f"(backbone lr={LR_BACKBONE}, head lr={LR})")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            if (step + 1) % LOG_EVERY_N_STEPS == 0:
                print(f"  Epoch {epoch} step {step+1} loss={loss.item():.4f}")

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        print(f"Epoch {epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            ckpt = CHECKPOINTS_DIR / "best.pt"
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_acc": val_acc},
                ckpt,
            )
            print(f"  Saved best checkpoint to {ckpt}")
        else:
            patience_counter += 1

        if (epoch % SAVE_EVERY_N_EPOCHS == 0) and epoch != best_epoch:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_acc": val_acc},
                CHECKPOINTS_DIR / f"epoch_{epoch}.pt",
            )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    print(f"Done. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
