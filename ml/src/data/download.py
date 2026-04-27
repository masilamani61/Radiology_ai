"""
ml/src/data/download.py
========================
Downloads chest X-ray datasets from Kaggle if not already present.
Verifies data integrity after download.

Datasets:
  1. Chest X-Ray Images (Pneumonia) — paultimothymooney
  2. COVID-19 Radiography Database  — tawsifurrahman

Usage:
    python ml/src/data/download.py
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name"   : "chest-xray-pneumonia",
        "kaggle" : "paultimothymooney/chest-xray-pneumonia",
        "verify" : RAW_DIR / "chest_xray/train/NORMAL",
        "desc"   : "Chest X-Ray (Normal + Pneumonia)",
    },
    {
        "name"   : "covid19-radiography",
        "kaggle" : "tawsifurrahman/covid19-radiography-database",
        "verify" : RAW_DIR / "COVID-19_Radiography_Dataset/COVID/images",
        "desc"   : "COVID-19 Radiography Database",
    },
]


def check_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    custom_dir  = os.environ.get("KAGGLE_CONFIG_DIR", "")

    if kaggle_json.exists():
        logger.info("Kaggle credentials found at ~/.kaggle/kaggle.json")
        return True

    if custom_dir and (Path(custom_dir) / "kaggle.json").exists():
        logger.info(f"Kaggle credentials found at {custom_dir}/kaggle.json")
        return True

    logger.error(
        "Kaggle credentials not found!\n"
        "  1. Go to https://www.kaggle.com/settings\n"
        "  2. Click 'Create New API Token'\n"
        "  3. Save kaggle.json to ~/.kaggle/\n"
        "  4. Run: chmod 600 ~/.kaggle/kaggle.json\n"
        "  OR set KAGGLE_CONFIG_DIR to folder containing kaggle.json"
    )
    return False


def count_images(folder: Path) -> int:
    """Count images in a folder recursively."""
    if not folder.exists():
        return 0
    exts = ["*.jpg", "*.jpeg", "*.png", "*.PNG", "*.JPG", "*.JPEG"]
    total = 0
    for ext in exts:
        total += len(list(folder.glob(ext)))
    return total


def download_dataset(dataset: dict) -> bool:
    """
    Download a Kaggle dataset if not already present.

    Args:
        dataset: dict with name, kaggle slug, verify path, desc

    Returns:
        True if dataset is available, False if download failed
    """
    verify_path = dataset["verify"]

    # Check if already downloaded
    if verify_path.exists() and count_images(verify_path) > 0:
        n = count_images(verify_path)
        logger.info(f"Already exists: {dataset['desc']} ({n} images)")
        return True

    logger.info(f"Downloading: {dataset['desc']}...")
    logger.info(f"  Kaggle slug: {dataset['kaggle']}")
    logger.info(f"  Destination: {RAW_DIR}")

    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", dataset["kaggle"],
                "-p", str(RAW_DIR),
                "--unzip",
            ],
            capture_output = False,
            text           = True,
            timeout        = 600,   # 10 min timeout
        )

        if result.returncode != 0:
            logger.error(f"Download failed for {dataset['name']}")
            return False

        # verify after download
        if verify_path.exists():
            n = count_images(verify_path)
            logger.info(f"Download complete: {n} images at {verify_path}")
            return True
        else:
            logger.error(f"Download succeeded but verify path not found: {verify_path}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out for {dataset['name']}")
        return False
    except FileNotFoundError:
        logger.error("kaggle CLI not found — run: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def verify_all() -> dict:
    """
    Verify all datasets and return counts.

    Returns:
        Dict mapping class name to image count
    """
    CLASS_PATHS = {
        "Normal"   : RAW_DIR / "chest_xray/train/NORMAL",
        "Pneumonia": RAW_DIR / "chest_xray/train/PNEUMONIA",
        "COVID19"  : RAW_DIR / "COVID-19_Radiography_Dataset/COVID/images",
    }

    logger.info("\nDataset Verification:")
    logger.info("-" * 40)
    counts  = {}
    all_ok  = True

    for cls, path in CLASS_PATHS.items():
        n = count_images(path)
        if n > 0:
            logger.info(f"  {cls:<12}: {n:>5} images  FOUND")
            counts[cls] = n
        else:
            logger.error(f"  {cls:<12}: NOT FOUND at {path}")
            all_ok = False

    logger.info("-" * 40)
    if all_ok:
        total = sum(counts.values())
        logger.info(f"  {'TOTAL':<12}: {total:>5} images")
        logger.info("All datasets verified successfully!")
    else:
        logger.error("Some datasets missing!")

    return counts


def main():
    logger.info("=" * 50)
    logger.info("RadiologyAI — Dataset Download & Verification")
    logger.info("=" * 50)

    # Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)

    # Download each dataset
    all_ok = True
    for ds in DATASETS:
        ok = download_dataset(ds)
        if not ok:
            all_ok = False

    # Final verification
    counts = verify_all()

    if not all_ok or not counts:
        logger.error("Dataset download incomplete!")
        sys.exit(1)

    logger.info("\nAll datasets ready for training!")


if __name__ == "__main__":
    main()
