"""Two-stream ResNet for pre+post damage classification."""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_two_stream_model(num_classes: int = 4, freeze_backbone: bool = False, dropout: float = 0.5):
    """
    Two ResNet-50 branches (pre, post), concatenate features, classify.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    backbone_pre = resnet50(weights=weights)
    backbone_post = resnet50(weights=weights)
    feat_dim = backbone_pre.fc.in_features
    backbone_pre.fc = nn.Identity()
    backbone_post.fc = nn.Identity()
    if freeze_backbone:
        for p in backbone_pre.parameters():
            p.requires_grad = False
        for p in backbone_post.parameters():
            p.requires_grad = False
    classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(feat_dim * 2, num_classes),
    )
    model = TwoStreamResNet(backbone_pre, backbone_post, classifier)
    return model


class TwoStreamResNet(nn.Module):
    def __init__(self, backbone_pre, backbone_post, classifier):
        super().__init__()
        self.backbone_pre = backbone_pre
        self.backbone_post = backbone_post
        self.classifier = classifier

    def forward(self, pre, post):
        f_pre = self.backbone_pre(pre)
        f_post = self.backbone_post(post)
        fused = torch.cat([f_pre, f_post], dim=1)
        return self.classifier(fused)
