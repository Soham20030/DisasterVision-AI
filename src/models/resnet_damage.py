"""
ResNet-50 backbone + 4-class damage head for transfer learning.

Uses torchvision pretrained weights; replaces the final fc layer for NUM_CLASSES.
Optional freeze of backbone for first N epochs to avoid overwriting pretrained features.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_damage_model(
    num_classes: int = 4,
    freeze_backbone: bool = False,
    pretrained: bool = True,
    dropout: float = 0.4,
) -> nn.Module:
    """
    Returns a ResNet-50 with final layer replaced for num_classes (damage levels).
    freeze_backbone: if True, only the new head is trained.
    dropout: dropout rate before fc (reduces overfitting).
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = resnet50(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.fc.parameters():
            p.requires_grad = True
    return backbone


def unfreeze_backbone(model: nn.Module) -> None:
    """Call after N epochs to fine-tune the full network."""
    for p in model.parameters():
        p.requires_grad = True
