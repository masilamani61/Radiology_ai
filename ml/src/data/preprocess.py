import json, logging
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[3]
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

IMAGE_SIZE  = (224, 224)
RANDOM_SEED = 42

CLASS_SOURCES = {
    "Normal"   : RAW_DIR / "chest_xray/train/NORMAL",
    "Pneumonia": RAW_DIR / "chest_xray/train/PNEUMONIA",
    "COVID19"  : RAW_DIR / "COVID-19_Radiography_Dataset/COVID/images",
}

def get_images(folder):
    exts = ["*.jpg","*.jpeg","*.png","*.PNG","*.JPEG","*.JPG"]
    imgs = []
    for e in exts:
        imgs.extend(folder.glob(e))
    return sorted(imgs)

def resize_save(src, dst):
    try:
        with Image.open(src) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, "JPEG", quality=90)
        return True
    except Exception as e:
        logger.warning(f"Failed {src.name}: {e}")
        return False

def main():
    stats = {}
    for cls, src_dir in CLASS_SOURCES.items():
        logger.info(f"\nProcessing: {cls}")
        if not src_dir.exists():
            logger.error(f"NOT FOUND: {src_dir}")
            continue
        paths = get_images(src_dir)
        logger.info(f"  Found {len(paths)} images")
        train_val, test = train_test_split(paths, test_size=0.15, random_state=RANDOM_SEED)
        train, val      = train_test_split(train_val, test_size=0.176, random_state=RANDOM_SEED)
        logger.info(f"  Split → train:{len(train)} val:{len(val)} test:{len(test)}")
        counts = {}
        for split_name, split_paths in [("train",train),("val",val),("test",test)]:
            out_dir = PROC_DIR / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            ok = 0
            for i, p in enumerate(tqdm(split_paths, desc=f"  {split_name}/{cls}", leave=False)):
                dst = out_dir / f"{cls}_{split_name}_{i:05d}.jpg"
                if resize_save(p, dst):
                    ok += 1
            counts[split_name] = ok
            logger.info(f"  {split_name}: {ok} saved")
        stats[cls] = counts

    json.dump(stats, open(PROC_DIR/"split_stats.json","w"), indent=2)
    logger.info("\nDone! split_stats.json saved.")
    logger.info("\nFinal counts:")
    for cls, c in stats.items():
        logger.info(f"  {cls}: train={c.get('train',0)} val={c.get('val',0)} test={c.get('test',0)}")

if __name__ == "__main__":
    main()
