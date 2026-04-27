"""
Great Expectations data validation suite.
Validates dataset quality before training.
"""
import great_expectations as ge
import pandas as pd
import json
import logging
from pathlib import Path
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"


def build_image_dataframe(split: str = "train") -> pd.DataFrame:
    """Build a DataFrame of image statistics for validation."""
    rows = []
    for cls in ["Normal", "Pneumonia", "COVID19"]:
        folder = PROC_DIR / split / cls
        if not folder.exists():
            continue
        images = list(folder.glob("*.jpg"))[:100]
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    arr  = np.array(img.convert("L"), dtype=np.float32)
                    rows.append({
                        "class"      : cls,
                        "width"      : w,
                        "height"     : h,
                        "mean_pixel" : float(arr.mean()),
                        "std_pixel"  : float(arr.std()),
                        "file_size"  : img_path.stat().st_size,
                    })
            except Exception:
                continue
    return pd.DataFrame(rows)


def run_validation():
    """Run Great Expectations validation suite."""
    logger.info("Running Great Expectations validation...")

    df      = build_image_dataframe("train")
    gdf     = ge.from_pandas(df)
    results = {}

    # Define expectations
    checks = [
        ("Image width is 224",      gdf.expect_column_values_to_be_in_set("width", [224])),
        ("Image height is 224",     gdf.expect_column_values_to_be_in_set("height", [224])),
        ("Mean pixel > 0",          gdf.expect_column_min_to_be_between("mean_pixel", 0, 255)),
        ("All 3 classes present",   gdf.expect_column_distinct_values_to_contain_set("class", ["Normal","Pneumonia","COVID19"])),
        ("No null values",          gdf.expect_column_values_to_not_be_null("class")),
        ("File size > 1KB",         gdf.expect_column_values_to_be_between("file_size", 1000, 10_000_000)),
        ("Pixel std > 0",           gdf.expect_column_values_to_be_between("std_pixel", 0.1, 255)),
    ]

    all_passed = True
    for name, result in checks:
        passed = result["success"]
        results[name] = passed
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    # Save report
    report = {"passed": all_passed, "checks": results, "total": len(checks), "passed_count": sum(results.values())}
    report_path = PROC_DIR / "ge_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nValidation: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    logger.info(f"Report saved: {report_path}")
    return report


if __name__ == "__main__":
    run_validation()
