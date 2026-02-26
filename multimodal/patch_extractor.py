"""Extract pre+post patches for multimodal damage classification."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def _center_crop_bbox(xmin, ymin, xmax, ymax, patch_size):
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    half = patch_size // 2
    return cx - half, cy - half, cx - half + patch_size, cy - half + patch_size


def _get_pre_path(post_path: Path) -> Path:
    name = post_path.name
    for old, new in [("post_disaster", "pre_disaster"), ("post-disaster", "pre-disaster")]:
        if old in name.lower():
            pre_name = name.replace(old, new) if old in name else name.replace(old.upper(), new)
            return post_path.parent / pre_name
    return post_path.parent / (post_path.stem.replace("post", "pre") + post_path.suffix)


def extract_patches_multimodal(
    post_path: Path,
    building_list: List[Tuple[Tuple[int, int, int, int], int]],
    processed_dir: Path,
    patch_size: int,
    prefix: str = "",
) -> List[Tuple[str, str, int]]:
    """
    Extract matching pre+post patches. Returns [(path_pre, path_post, damage), ...].
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    pre_path = _get_pre_path(post_path)
    if not pre_path.is_file():
        return []
    img_pre = np.array(Image.open(pre_path).convert("RGB"))
    img_post = np.array(Image.open(post_path).convert("RGB"))
    h_pre, w_pre = img_pre.shape[:2]
    h_post, w_post = img_post.shape[:2]
    base = post_path.stem.replace("_post_disaster", "").replace("_post-disaster", "")
    if prefix:
        base = f"{prefix}_{base}"
    out = []
    for i, (bbox, damage) in enumerate(building_list):
        xmin, ymin, xmax, ymax = bbox
        left, top, right, bottom = _center_crop_bbox(xmin, ymin, xmax, ymax, patch_size)
        for img, h, w, suffix in [(img_pre, h_pre, w_pre, "pre"), (img_post, h_post, w_post, "post")]:
            pl, pt = max(0, -left), max(0, -top)
            pr, pb = max(0, right - w), max(0, bottom - h)
            l, t, r, b = max(0, left), max(0, top), min(w, right), min(h, bottom)
            crop = img[t:b, l:r]
            if pl or pt or pr or pb:
                crop = np.pad(crop, ((pt, pb), (pl, pr), (0, 0)), mode="constant", constant_values=0)
            if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
                crop = np.array(Image.fromarray(crop).resize((patch_size, patch_size)))
            rel = f"{base}_{i:04d}_{suffix}.png"
            Image.fromarray(crop).save(processed_dir / rel)
        out.append((f"{base}_{i:04d}_pre.png", f"{base}_{i:04d}_post.png", damage))
    return out
