"""
Extract xBD zip, prepare multimodal (pre+post) patches, optionally delete raw.

Usage:
  python scripts/prepare_multimodal_from_zip.py [path/to/xbd.zip]
  python scripts/prepare_multimodal_from_zip.py --keep-raw
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.multimodal import DATA_DIR, PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", nargs="?", default=DATA_DIR / "xbd.zip", type=Path)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--remove-zip", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.is_file():
        print(f"Zip not found: {zip_path}")
        sys.exit(1)

    extract_dir = DATA_DIR / "xbd"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)

    inner = extract_dir / "xbd"
    if inner.is_dir() and (inner / "train").is_dir():
        for item in inner.iterdir():
            dest = extract_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(item), str(extract_dir))
        inner.rmdir()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Running multimodal prepare_subset...")
    from multimodal.prepare_subset import main as run_prepare
    run_prepare()

    if not args.keep_raw:
        shutil.rmtree(extract_dir)
        print("Raw data removed.")
    if args.remove_zip:
        zip_path.unlink()
        print("Zip removed.")
    print("Done.")


if __name__ == "__main__":
    main()
