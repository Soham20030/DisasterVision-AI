"""
Grad-CAM: gradient-weighted class activation mapping.

Uses the last conv layer (layer4) activations and gradients of the target class
to produce a heatmap showing which regions influenced the prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    """
    Grad-CAM for ResNet. Hooks layer4 to capture activations,
    then uses gradients of the target class score to weight channels.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _register_hooks(self):
        self._hooks = [
            self.target_layer.register_forward_hook(self._save_activations),
            self.target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given input.
        target_class: class to explain (default: predicted class).
        Returns: 2D heatmap (H, W) in [0, 1].
        """
        self.model.eval()
        self._register_hooks()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()

        acts = self.activations[0].cpu().numpy()
        grads = self.gradients[0].cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        self._remove_hooks()
        return cam.astype(np.float32)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay heatmap on image. image: (H,W,3) uint8, heatmap: (h,w) float.
    Resizes heatmap to image size and blends.
    """
    import matplotlib.pyplot as plt
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR
        )
    ) / 255.0
    heatmap_colored = plt.get_cmap(colormap)(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay
