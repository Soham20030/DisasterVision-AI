"""
Quick check of raw/ or train data: layout, disaster types, and one label sample.
Run from project root: python scripts/check_data.py
"""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

def main():
    # Prefer flat layout (Kaggle): data/train/train/
    train_img = DATA / "train" / "train" / "images"
    train_lb = DATA / "train" / "train" / "labels"
    raw = DATA / "raw"

    if train_img.is_dir() and train_lb.is_dir():
        print("Layout: flat (data/train/train/images + labels)")
        imgs = list(train_img.glob("*_post_disaster.png"))
        labels = list(train_lb.glob("*_post_disaster.json"))
        disasters = sorted({f.name.split("_")[0] for f in imgs})
        print("Disaster prefixes:", disasters)
        print("Post-disaster images:", len(imgs), "| Post-disaster labels:", len(labels))
        # Sample one label
        sample = train_lb / (imgs[0].stem + ".json") if imgs else None
        if sample and sample.exists():
            sample = train_lb / list(train_lb.glob("*_post_disaster.json"))[0]
            with open(sample) as f:
                d = json.load(f)
            feats = d.get("features", {})
            if isinstance(feats, dict) and "xy" in feats:
                print("Label format: features.xy[] with .properties.subtype and .wkt")
                n_buildings = len(feats["xy"])
                print("Sample file", sample.name, "->", n_buildings, "buildings")
                if feats["xy"]:
                    first = feats["xy"][0]
                    print("  First building subtype:", first.get("properties", {}).get("subtype"))
            else:
                print("Label format: unknown (features structure:", type(feats), ")")
    elif raw.is_dir():
        subs = [s.name for s in raw.iterdir() if s.is_dir()]
        print("Layout: data/raw with subdirs:", subs or "(none)")
    else:
        print("No train/train or raw found under", DATA)


if __name__ == "__main__":
    main()
