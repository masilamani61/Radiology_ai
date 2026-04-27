"""
Drift detection using Evidently AI.
Compares current data pixel statistics vs training baseline.
Generates HTML report saved to data/processed/drift_report.html
"""
import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"


def compute_image_stats(folder: Path, sample_size: int = 200) -> pd.DataFrame:
    """
    Compute pixel statistics for images in a folder.
    Returns DataFrame with one row per image.
    """
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(list(folder.glob(ext))[:sample_size])

    rows = []
    for img_path in tqdm(images, desc=f"  {folder.name}", leave=False):
        try:
            arr = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
            rows.append({
                "mean_pixel"    : float(arr.mean()),
                "std_pixel"     : float(arr.std()),
                "min_pixel"     : float(arr.min()),
                "max_pixel"     : float(arr.max()),
                "brightness"    : float(arr.mean() / 255.0),
                "contrast"      : float(arr.std() / 255.0),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def run_drift_detection() -> dict:
    """
    Run Evidently drift detection between baseline and current data.
    Returns drift summary dict.
    """
    logger.info("Running Evidently AI drift detection...")

    results = {}

    for cls in ["Normal", "Pneumonia", "COVID19"]:
        train_dir = PROC_DIR / "train" / cls
        test_dir  = PROC_DIR / "test"  / cls

        if not train_dir.exists() or not test_dir.exists():
            logger.warning(f"Skipping {cls} — folder not found")
            continue

        logger.info(f"Analysing class: {cls}")

        # Reference = training data (baseline)
        # Current   = test data (simulates new incoming data)
        reference = compute_image_stats(train_dir, sample_size=200)
        current   = compute_image_stats(test_dir,  sample_size=100)

        if reference.empty or current.empty:
            continue

        # Run Evidently report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        report.run(reference_data=reference, current_data=current)

        # Save HTML report
        report_path = PROC_DIR / f"drift_report_{cls}.html"
        report.save_html(str(report_path))
        logger.info(f"Drift report saved: {report_path}")

        # Extract drift summary
        report_dict  = report.as_dict()
        drift_score  = report_dict["metrics"][0]["result"].get("drift_share", 0)
        drift_detected = drift_score > 0.3

        results[cls] = {
            "drift_detected" : drift_detected,
            "drift_score"    : round(drift_score, 4),
            "report_path"    : str(report_path),
        }

        logger.info(
            f"  {cls}: drift_score={drift_score:.3f} "
            f"drift_detected={drift_detected}"
        )

    # Save summary
    summary_path = PROC_DIR / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Drift summary saved: {summary_path}")
    any_drift = any(v["drift_detected"] for v in results.values())
    logger.info(f"Overall drift detected: {any_drift}")

    return results


if __name__ == "__main__":
    results = run_drift_detection()
    print("\nDrift Detection Results:")
    for cls, r in results.items():
        status = "DRIFT DETECTED" if r["drift_detected"] else "No drift"
        print(f"  {cls:<12}: {status} (score={r['drift_score']:.3f})")
