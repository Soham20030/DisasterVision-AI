"""
Inference wrapper for the Two-Stream ResNet-50 multimodal damage classifier.

Wraps the existing trained model safely for batch inference.
Falls back to a realistic mock if the model cannot be loaded.
"""

import sys
import logging
import io
import base64
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
INFERENCE_VERSION = "1.0.2-gradcam"

# ── Resolve project root so we can import the existing model code ─────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # disaster_vision/backend/ → project root
sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["no-damage", "minor", "major", "destroyed"]
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints_multimodal" / "best.pt"

# ── Attempt to load models ────────────────────────────────────────────────────
_multi_model = None
_uni_model = None
_device = None
_transform = None
_use_mock = False

UNIMODAL_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
_multi_gradcam = None
_uni_gradcam = None
_load_lock = threading.Lock()


def _load_models():
    """Load real models once. Sets _use_mock on failure."""
    global _multi_model, _uni_model, _device, _transform, _use_mock, _multi_gradcam, _uni_gradcam
    
    with _load_lock:
        if _multi_model is not None or _uni_model is not None or _use_mock:
            return

        logger.info("--- MODEL LOADING START ---")
    try:
        import torch
        import torchvision.transforms as T
        from multimodal.model import build_two_stream_model
        from src.models.resnet_damage import build_damage_model

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # Load Multimodal
        try:
            print(f"Loading multimodal from {CHECKPOINT_PATH}...")
            _multi_model = build_two_stream_model(num_classes=4, freeze_backbone=False, dropout=0.5)
            ckpt_multi = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=True)
            _multi_model.load_state_dict(ckpt_multi["model_state_dict"])
            _multi_model = _multi_model.to(_device).eval()
            print(f"SUCCESS: Multimodal model loaded.")
        except Exception as e:
            print(f"ERROR: Multimodal load failed: {e}")

        # Load Unimodal
        try:
            print(f"Loading unimodal from {UNIMODAL_CHECKPOINT}...")
            _uni_model = build_damage_model(num_classes=4, pretrained=False)
            ckpt_uni = torch.load(UNIMODAL_CHECKPOINT, map_location=_device, weights_only=True)
            state_dict = ckpt_uni.get("model_state_dict", ckpt_uni)
            _uni_model.load_state_dict(state_dict)
            _uni_model = _uni_model.to(_device).eval()
            print(f"SUCCESS: Unimodal model loaded.")
        except Exception as e:
            print(f"ERROR: Unimodal load failed: {e}")

        # Initialize Grad-CAM
        if not _use_mock:
            try:
                from src.visualization.gradcam import GradCAM
                if _multi_model is not None:
                    _multi_gradcam = GradCAM(_multi_model, _multi_model.backbone_post.layer4)
                if _uni_model is not None:
                    _uni_gradcam = GradCAM(_uni_model, _uni_model.layer4)
                print("SUCCESS: Grad-CAM initialized.")
            except Exception as e:
                print(f"WARNING: Grad-CAM initialization failed: {e}")

        _use_mock = (_multi_model is None and _uni_model is None)
        if _use_mock:
             logger.warning("Both models failed to load. Using mock mode.")

    except Exception as e:
        logger.error(f"CRITICAL: Core load failure: {e}")
        _use_mock = True
    logger.info("--- MODEL LOADING END ---")


def _real_predict(pre_img: Optional[Image.Image], post_img: Image.Image) -> dict:
    """Run inference using the appropriate loaded model."""
    import torch

    # Run inference with gradients ENABLED for Grad-CAM
    # We use a separate context since Grad-CAM needs backprop
    if pre_img is not None and _multi_model is not None:
        # Dual-image mode
        pre_t = _transform(pre_img).unsqueeze(0).to(_device).requires_grad_(True)
        post_t = _transform(post_img).unsqueeze(0).to(_device).requires_grad_(True)
        logits = _multi_model(pre_t, post_t)
        gradcam_tool = _multi_gradcam
        # For multimodal, we need a special call if we want to target specifically the post branch logic
        # But our GradCAM implementation handles the full model.backward() which is fine.
    elif _uni_model is not None:
        # Single-image mode
        post_t = _transform(post_img).unsqueeze(0).to(_device).requires_grad_(True)
        logits = _uni_model(post_t)
        gradcam_tool = _uni_gradcam
    else:
        return _mock_predict(pre_img, post_img)

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))

    # Generate Grad-CAM heatmaps
    gradcam_b64 = None
    if gradcam_tool is not None:
        try:
            logger.info(f"--- Grad-CAM START (tool: {gradcam_tool}, pred: {pred_idx}) ---")
            from src.visualization.gradcam import overlay_heatmap
            # Compute heatmap for the predicted class
            if pre_img is not None and _multi_model:
                heatmap = gradcam_tool(pre_t, post_t, target_class=pred_idx)
            else:
                heatmap = gradcam_tool(post_t, target_class=pred_idx)
            
            logger.info(f"Heatmap generated: {heatmap.shape}, max={heatmap.max():.4f}")
            
            # Overlay on original image (resized to 224 for consistency)
            img_np = np.array(post_img.resize((IMG_SIZE, IMG_SIZE)))
            overlay = overlay_heatmap(img_np, heatmap)
            
            # Convert to base64
            buffered = io.BytesIO()
            overlay_pil = Image.fromarray(overlay)
            overlay_pil.save(buffered, format="JPEG", quality=85)
            gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            logger.info(f"--- Grad-CAM END (b64 len: {len(gradcam_b64)}) ---")
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return {
        "class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)},
        "gradcam": gradcam_b64
    }


def _mock_predict(pre_img: Image.Image, post_img: Image.Image) -> dict:
    """
    Realistic mock: randomly picks a class with plausible confidence.
    Used when the real model is unavailable.
    """
    rng = np.random.default_rng()
    # Weighted toward damaged classes for demo realism
    weights = [0.25, 0.30, 0.28, 0.17]
    pred_idx = int(rng.choice(4, p=weights))
    # Generate plausible probability vector
    raw = rng.dirichlet(np.ones(4) * 0.5)
    # Boost predicted class
    raw[pred_idx] = raw[pred_idx] + 0.4
    raw = raw / raw.sum()

    return {
        "class": CLASS_NAMES[pred_idx],
        "confidence": float(raw[pred_idx]),
        "probabilities": {name: float(raw[i]) for i, name in enumerate(CLASS_NAMES)},
        "_mock": True,
    }


def predict_damage(pre_img: Optional[Image.Image], post_img: Image.Image) -> dict:
    """
    Public inference function. Supports both Dual and Single image analysis.

    Args:
        pre_img:  PIL Image of the pre-disaster tile (Optional for single-image mode).
        post_img: PIL Image of the post-disaster tile.

    Returns:
        Analysis result dictionary.
    """
    global _multi_model, _uni_model

    # Lazy-load models on first call
    if _multi_model is None and _uni_model is None and not _use_mock:
        _load_models()

    if _use_mock:
        return _mock_predict(pre_img, post_img)
    return _real_predict(pre_img, post_img)


def is_mock_mode() -> bool:
    """Returns True if the system is using mock inference (real model not loaded)."""
    return _use_mock
