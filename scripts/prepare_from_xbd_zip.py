"""
Extract xBD from zip, run patch extraction (50K samples), then delete raw data to save space.

Usage:
  python scripts/prepare_from_xbd_zip.py                    # use data/xbd.zip
  python scripts/prepare_from_xbd_zip.py path/to/xbd.zip   # custom zip path
  python scripts/prepare_from_xbd_zip.py --keep-raw        # don't delete raw after

Place xbd.zip in data/ (or specify path). Expects zip with train/, tier1/, tier3/, hold/
containing images/ and labels/ subfolders.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.default import DATA_DIR, PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Extract xBD zip, prepare 50K patches, optionally delete raw")
    parser.add_argument("zip_path", nargs="?", default=DATA_DIR / "xbd.zip", type=Path)
    parser.add_argument("--keep-raw", action="store_true", help="Keep extracted raw data after processing")
    parser.add_argument("--remove-zip", action="store_true", help="Also delete zip file after (saves ~33GB)")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.is_file():
        print(f"Zip not found: {zip_path}")
        print("Place xbd.zip in data/ or pass path: python scripts/prepare_from_xbd_zip.py path/to/xbd.zip")
        sys.exit(1)

    extract_dir = DATA_DIR / "xbd"
    if extract_dir.exists():
        print("Removing existing", extract_dir)
        shutil.rmtree(extract_dir)
    if PROCESSED_DIR.exists():
        print("Clearing", PROCESSED_DIR, "(fresh extraction)")
        shutil.rmtree(PROCESSED_DIR)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting", zip_path, "to", extract_dir)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)

    # Handle nested xbd/ in zip (some zips extract to xbd/xbd/...)
    inner = extract_dir / "xbd"
    if inner.is_dir() and (inner / "train").is_dir():
        for item in inner.iterdir():
            dest = extract_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(item), str(extract_dir))
        inner.rmdir()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Running prepare_subset...")
    from src.data.prepare_subset import main as run_prepare
    run_prepare()

    if not args.keep_raw:
        print("Removing raw data to free space...")
        shutil.rmtree(extract_dir)
        print("Raw data removed. Patches in data/processed/")

    if args.remove_zip:
        zip_path.unlink()
        print("Zip removed.")

    print("Done.")


if __name__ == "__main__":
    main()
