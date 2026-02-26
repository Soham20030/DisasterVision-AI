"""
Download xBD data and optionally Tier 3 for more training data.

Usage:
  python scripts/download_xbd.py           # main train+test only
  python scripts/download_xbd.py --tier3   # main + Tier 3 (more data)

Requires: pip install kaggle, ~/.kaggle/kaggle.json
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def _ensure_raw():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _run_kaggle_download(dataset_slug: str) -> Path:
    """Download and unzip a Kaggle dataset. Returns path to extracted folder."""
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-p", str(DATA_DIR),
                "--unzip",
                dataset_slug,
            ],
            check=True,
        )
    except FileNotFoundError:
        print("Kaggle CLI not found. Install: pip install kaggle")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print("Kaggle download failed. Check ~/.kaggle/kaggle.json")
        raise SystemExit(1) from e
    for d in DATA_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            if (d / "train").is_dir() or any((d / x).is_dir() for x in ("images", "labels")):
                return d
    return DATA_DIR


def download_from_kaggle():
    """Download main xView2 train+test dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _run_kaggle_download("tunguz/xview2-challenge-dataset-train-and-test")


def download_tier3() -> Path:
    """Download xView2 Tier 3 dataset. Returns path to extracted folder."""
    print("Downloading Tier 3...")
    return _run_kaggle_download("tunguz/xview2-challenge-dataset-tier-3-data")


def _find_train_images_labels(root: Path):
    """Find (images_dir, labels_dir) under root. Returns (Path, Path) or (None, None)."""
    candidates = [
        (root / "train" / "train" / "images", root / "train" / "train" / "labels"),
        (root / "train" / "images", root / "train" / "labels"),
        (root / "images", root / "labels"),
    ]
    for im_d, lb_d in candidates:
        if im_d.is_dir() and lb_d.is_dir():
            return im_d, lb_d
    # Some layouts have images/labels inside disaster subdirs
    train = root / "train"
    if train.is_dir():
        for sub in train.iterdir():
            if sub.is_dir():
                im_d, lb_d = sub / "images", sub / "labels"
                if im_d.is_dir() and lb_d.is_dir():
                    return im_d, lb_d
    return None, None


def merge_tier3_into_train(main_root: Path, tier3_root: Path):
    """Copy Tier 3 images and labels into main train folder. Prefix filenames to avoid collisions."""
    main_im, main_lb = _find_train_images_labels(main_root)
    tier3_im, tier3_lb = _find_train_images_labels(tier3_root)
    if not main_im or not main_lb:
        print("Could not find main train/images and train/labels")
        return
    if not tier3_im or not tier3_lb:
        print("Could not find Tier 3 images and labels")
        return
    prefix = "tier3_"
    n_im = sum(1 for f in tier3_im.iterdir() if f.suffix.lower() in (".png", ".jpg", ".tif", ".tiff"))
    n_lb = sum(1 for f in tier3_lb.iterdir() if f.suffix.lower() == ".json")
    for f in tier3_im.iterdir():
        if f.suffix.lower() in (".png", ".jpg", ".tif", ".tiff"):
            shutil.copy2(f, main_im / f"{prefix}{f.name}")
    for f in tier3_lb.iterdir():
        if f.suffix.lower() == ".json":
            shutil.copy2(f, main_lb / f"{prefix}{f.name}")
    print(f"Merged Tier 3: {n_im} images, {n_lb} labels -> {main_im.parent}")


def organize_into_raw(source_root: Path, disaster_types: list):
    """
    Copy only earthquake/tsunami events from source_root into RAW_DIR.
    Expects source_root to contain train/ (and maybe test/) with disaster-named subdirs,
    each having images/ and labels/ (or similar). Adapt if your download layout differs.
    """
    _ensure_raw()
    source = Path(source_root)
    # Common layouts: source/train/, or source/ is already train/
    train_dirs = [source / "train", source]
    if (source / "train").is_dir():
        train_dirs = [source / "train"]
    elif (source / "xview2-challenge-dataset-train-and-test").is_dir():
        train_dirs = [source / "xview2-challenge-dataset-train-and-test" / "train"]

    for train_dir in train_dirs:
        if not train_dir.is_dir():
            continue
        for event_dir in train_dir.iterdir():
            if not event_dir.is_dir():
                continue
            name_lower = event_dir.name.lower()
            if not any(d in name_lower for d in disaster_types):
                continue
            # Target: data/raw/{event_name}/images and labels
            dest = RAW_DIR / event_dir.name
            im_dest = dest / "images"
            lb_dest = dest / "labels"
            im_dest.mkdir(parents=True, exist_ok=True)
            lb_dest.mkdir(parents=True, exist_ok=True)
            # Copy: event_dir may have images/ and labels/ already, or files at top level
            for sub in ("images", "Images"):
                src_im = event_dir / sub
                if src_im.is_dir():
                    for f in src_im.iterdir():
                        if f.suffix.lower() in (".png", ".jpg", ".tif", ".tiff"):
                            shutil.copy2(f, im_dest / f.name)
                    break
            for sub in ("labels", "Labels", "label"):
                src_lb = event_dir / sub
                if src_lb.is_dir():
                    for f in src_lb.iterdir():
                        if f.suffix.lower() == ".json":
                            shutil.copy2(f, lb_dest / f.name)
                    break
            # If no images/labels subdirs, look for post_disaster.* and *.json at top level
            if not any(im_dest.iterdir()):
                for f in event_dir.iterdir():
                    if "post" in f.name.lower() and f.suffix.lower() in (".png", ".jpg"):
                        shutil.copy2(f, im_dest / f.name)
            if not any(lb_dest.iterdir()):
                for f in event_dir.iterdir():
                    if f.suffix.lower() == ".json":
                        shutil.copy2(f, lb_dest / f.name)
            if any(im_dest.iterdir()) and any(lb_dest.iterdir()):
                print("Organized:", event_dir.name, "->", dest)
    print("Raw data under:", RAW_DIR)


def main():
    parser = argparse.ArgumentParser(description="Download xBD and optionally Tier 3")
    parser.add_argument("--zip", type=Path, help="Path to existing xView2 zip")
    parser.add_argument("--source", type=Path, help="Path to already-unzipped xView2 root")
    parser.add_argument("--tier3", action="store_true", help="Also download and merge Tier 3 data")
    parser.add_argument("--disasters", nargs="+", default=["earthquake", "tsunami", "hurricane", "flood", "volcano", "fire"])
    args = parser.parse_args()

    source_root = None
    if args.zip:
        dest_dir = DATA_DIR / args.zip.stem
        dest_dir.mkdir(parents=True, exist_ok=True)
        print("Unzipping to", dest_dir)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(dest_dir)
        source_root = dest_dir
    elif args.source:
        source_root = args.source
    else:
        try:
            source_root = Path(download_from_kaggle())
        except SystemExit:
            raise

    if args.tier3:
        tier3_root = Path(download_tier3())
        merge_tier3_into_train(source_root, tier3_root)

    organize_into_raw(source_root, args.disasters)


if __name__ == "__main__":
    main()
