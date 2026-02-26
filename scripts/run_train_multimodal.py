"""
Train two-stream (pre+post) damage model.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.multimodal import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CLASS_WEIGHT_MAX,
    DROPOUT_RATE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    FREEZE_BACKBONE_EPOCHS,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LR,
    LR_BACKBONE,
    MANIFEST_PATH,
    NUM_CLASSES,
    PROCESSED_DIR,
    WEIGHT_DECAY,
)
from multimodal.loaders import get_multimodal_loaders
from multimodal.model import build_two_stream_model


def unfreeze_backbones(model):
    for p in model.parameters():
        p.requires_grad = True


def main():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    loaders = get_multimodal_loaders(MANIFEST_PATH, PROCESSED_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    train_loader, val_loader = loaders["train"], loaders["val"]

    import pandas as pd
    manifest = pd.read_csv(MANIFEST_PATH)
    train_labels = manifest[manifest["split"] == "train"]["label"].values
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    weights = np.minimum(weights, CLASS_WEIGHT_MAX)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = build_two_stream_model(num_classes=NUM_CLASSES, freeze_backbone=(FREEZE_BACKBONE_EPOCHS > 0), dropout=DROPOUT_RATE)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_BACKBONE_EPOCHS + 1 and FREEZE_BACKBONE_EPOCHS > 0:
            unfreeze_backbones(model)
            bp = [p for n, p in model.named_parameters() if "backbone" in n]
            hp = [p for n, p in model.named_parameters() if "classifier" in n]
            optimizer = torch.optim.AdamW(
                [{"params": bp, "lr": LR_BACKBONE}, {"params": hp, "lr": LR}],
                weight_decay=WEIGHT_DECAY,
            )
            print("Unfroze backbones at epoch", epoch)

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for step, (pre, post, labels) in enumerate(train_loader):
            pre, post, labels = pre.to(device), post.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pre, post)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch} step {step+1} loss={loss.item():.4f}")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for pre, post, labels in val_loader:
                pre, post, labels = pre.to(device), post.to(device), labels.to(device)
                logits = model(pre, post)
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch} train_loss={train_loss/len(train_loader):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, CHECKPOINTS_DIR / "best.pt")
            print("  Saved best checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("Early stopping at epoch", epoch)
                break

    print("Training complete. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
