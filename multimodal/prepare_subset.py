"""
Prepare multimodal (pre+post) patches from xBD. Run after extracting xbd.zip.
Creates data/processed_multimodal/ with manifest (path_pre, path_post, label, split).
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.multimodal import (
    DATA_DIR,
    DISASTER_TYPES,
    MAX_NO_DAMAGE_SAMPLES,
    MAX_SAMPLES_PER_DISASTER,
    MAX_TOTAL_SAMPLES,
    PROCESSED_DIR,
    PATCH_SIZE,
    RANDOM_SEED,
    VAL_RATIO,
    TEST_RATIO,
)
from src.data.xbd_parser import find_disaster_pairs, find_pairs_flat, parse_label_file
from multimodal.patch_extractor import extract_patches_multimodal


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_disaster_pairs(DATA_DIR / "raw", DISASTER_TYPES)
    if not pairs:
        candidates = []
        flat_im = DATA_DIR / "train" / "train" / "images"
        flat_lb = DATA_DIR / "train" / "train" / "labels"
        if flat_im.is_dir() and flat_lb.is_dir():
            candidates.append((flat_im, flat_lb))
        for sub in ("tier1", "tier3"):
            im, lb = DATA_DIR / sub / "images", DATA_DIR / sub / "labels"
            if im.is_dir() and lb.is_dir():
                candidates.append((im, lb))
        xbd_root = DATA_DIR / "xbd"
        for sub in ("train", "tier1", "tier3", "hold"):
            im, lb = xbd_root / sub / "images", xbd_root / sub / "labels"
            if im.is_dir() and lb.is_dir():
                candidates.append((im, lb))
        for flat_im, flat_lb in candidates:
            pairs.extend(find_pairs_flat(flat_im, flat_lb, DISASTER_TYPES))

    if not pairs:
        print("No pairs found. Extract xbd.zip first: python scripts/prepare_multimodal_from_zip.py")
        return

    all_rows = []
    per_disaster = {}
    for im_path, lb_path in pairs:
        parts = im_path.name.split("_")
        disaster_name = parts[1] if parts[0].lower() in ("tier1", "tier3") and len(parts) >= 2 else parts[0]
        building_list = parse_label_file(lb_path)
        if not building_list:
            continue
        n_before = per_disaster.get(disaster_name, 0)
        cap = MAX_SAMPLES_PER_DISASTER - n_before
        if cap <= 0:
            continue
        if len(building_list) > cap:
            import random
            rng = random.Random(RANDOM_SEED)
            building_list = rng.sample(building_list, cap)
        extracted = extract_patches_multimodal(
            im_path, building_list, PROCESSED_DIR, PATCH_SIZE, prefix=disaster_name[:20]
        )
        for path_pre, path_post, label in extracted:
            all_rows.append({"path_pre": path_pre, "path_post": path_post, "label": label, "disaster": disaster_name})
        per_disaster[disaster_name] = per_disaster.get(disaster_name, 0) + len(extracted)

    print("Extracted", len(all_rows), "pre+post pairs before caps.")
    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    if MAX_NO_DAMAGE_SAMPLES is not None:
        no_dmg = df[df["label"] == 0]
        if len(no_dmg) > MAX_NO_DAMAGE_SAMPLES:
            rng = __import__("random").Random(RANDOM_SEED)
            keep = rng.sample(no_dmg.index.tolist(), MAX_NO_DAMAGE_SAMPLES)
            df = df.drop(no_dmg.index.difference(keep)).reset_index(drop=True)
            print("Capped no-damage to", MAX_NO_DAMAGE_SAMPLES, ". Total:", len(df))
    if MAX_TOTAL_SAMPLES is not None and len(df) > MAX_TOTAL_SAMPLES:
        df = df.sample(n=MAX_TOTAL_SAMPLES, random_state=RANDOM_SEED, stratify=df["label"]).reset_index(drop=True)
        print("Stratified sample to", len(df))

    min_per = df["label"].value_counts().min()
    strat = min_per >= 2
    train_df, rest = train_test_split(df, test_size=VAL_RATIO + TEST_RATIO, stratify=df["label"] if strat else None, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(rest, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), stratify=rest["label"] if strat and rest["label"].value_counts().min() >= 2 else None, random_state=RANDOM_SEED)
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest.to_csv(PROCESSED_DIR / "manifest.csv", index=False)
    print("Manifest saved:", PROCESSED_DIR / "manifest.csv")
    print("Label counts:\n", manifest["label"].value_counts().sort_index())
    print("Split counts:\n", manifest["split"].value_counts())
