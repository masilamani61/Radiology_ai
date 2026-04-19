import json, logging, sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR= ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

CLASS_FOLDERS = {
    "Normal"   : RAW_DIR / "chest_xray/train/NORMAL",
    "Pneumonia": RAW_DIR / "chest_xray/train/PNEUMONIA",
    "COVID19"  : RAW_DIR / "COVID-19_Radiography_Dataset/COVID/images",
}

def validate():
    results = {}
    baseline = {}
    for cls, folder in CLASS_FOLDERS.items():
        if not folder.exists():
            logger.error(f"NOT FOUND: {folder}")
            continue
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")) + list(folder.glob("*.PNG"))
        valid, corrupt = 0, 0
        for p in tqdm(imgs, desc=f"  {cls}", leave=False):
            try:
                Image.open(p).verify()
                valid += 1
            except:
                corrupt += 1
        results[cls] = {"total": len(imgs), "valid": valid, "corrupt": corrupt}
        logger.info(f"  {cls}: {len(imgs)} images, {corrupt} corrupt")
        import numpy as np
        sample = imgs[:200]
        means = []
        for p in sample:
            try:
                arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
                means.append(float(arr.mean()))
            except: pass
        baseline[cls] = {"pixel_mean": round(float(np.mean(means)),4), "pixel_std": round(float(np.std(means)),4), "sample_size": len(means)}

    json.dump(results,  open(PROC_DIR/"validation_report.json","w"), indent=2)
    json.dump(baseline, open(PROC_DIR/"baseline_stats.json","w"),    indent=2)
    logger.info("Saved validation_report.json and baseline_stats.json")

if __name__ == "__main__":
    validate()
