"""
ml/src/evaluation/evaluate.py
==============================
Comprehensive model evaluation on the held-out test set.
Logs all metrics to MLflow and saves artefacts for DVC.

Metrics computed:
  - Accuracy, Macro F1, Macro AUC-ROC
  - Per-class: Precision, Recall, F1, AUC
  - Sensitivity (Recall) — critical for clinical setting
  - Specificity per class
  - Confusion matrix PNG

Usage:
    python ml/src/evaluation/evaluate.py
    python ml/src/evaluation/evaluate.py --model_path ml/models/efficientnetb0_best.pth
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from ml.src.data.dataloader import CLASS_NAMES, get_dataloaders
from ml.src.models.model import get_model
from shared.config import get_config_value, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[3]
PROC_DIR   = resolve_path(get_config_value("paths", "processed_data_dir", default="data/processed"))
MODELS_DIR = resolve_path(get_config_value("paths", "models_dir", default="ml/models"))
EXP_DIR    = resolve_path(get_config_value("paths", "experiments_dir", default="ml/experiments"))
EXP_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_EXPERIMENT_NAME = get_config_value(
    "ops",
    "mlflow",
    "experiment_name",
    default="radiologyai_xray_classification",
)
ROC_TITLE = f"ROC Curves - {get_config_value('app', 'name', default='RadiologyAI')} Chest X-Ray Classifier"
ACCURACY_MIN = get_config_value("ml", "evaluation", "acceptance", "accuracy_min", default=0.92)
MACRO_F1_MIN = get_config_value("ml", "evaluation", "acceptance", "macro_f1_min", default=0.92)
MACRO_AUC_MIN = get_config_value("ml", "evaluation", "acceptance", "macro_auc_min", default=0.95)
RECALL_MIN_PER_CLASS = get_config_value(
    "ml",
    "evaluation",
    "acceptance",
    "recall_min_per_class",
    default=0.9,
)


@torch.no_grad()
def get_predictions(
    model  : torch.nn.Module,
    loader : torch.utils.data.DataLoader,
    device : torch.device,
):
    """
    Run inference on entire dataset split.

    Args:
        model:  Trained model in eval mode
        loader: DataLoader for the split
        device: torch.device

    Returns:
        (all_labels, all_preds, all_probs)
        - all_labels: true class indices [N]
        - all_preds:  predicted class indices [N]
        - all_probs:  softmax probabilities [N, C]
    """
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_specificity(
    labels     : np.ndarray,
    preds      : np.ndarray,
    class_idx  : int,
) -> float:
    """
    Compute specificity (true negative rate) for one class.
    Specificity = TN / (TN + FP)

    Args:
        labels:    True labels
        preds:     Predicted labels
        class_idx: Class to compute specificity for

    Returns:
        Specificity float [0, 1]
    """
    binary_labels = (labels == class_idx).astype(int)
    binary_preds  = (preds  == class_idx).astype(int)
    tn = ((binary_labels == 0) & (binary_preds == 0)).sum()
    fp = ((binary_labels == 0) & (binary_preds == 1)).sum()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def save_roc_curves(
    labels    : np.ndarray,
    probs     : np.ndarray,
    save_path : Path,
) -> None:
    """
    Save multi-class ROC curve plot as PNG artefact.

    Args:
        labels:    True labels
        probs:     Softmax probabilities [N, C]
        save_path: Output PNG path
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["#378ADD", "#E24B4A", "#EF9F27"]

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        binary = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(ROC_TITLE)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved: {save_path}")


def save_confusion_matrix(
    labels    : np.ndarray,
    preds     : np.ndarray,
    save_path : Path,
) -> None:
    """Save confusion matrix PNG."""
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im  = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.colorbar(im, ax=ax)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(args: argparse.Namespace) -> dict:
    """
    Full evaluation pipeline — loads model, runs test set,
    logs all metrics to MLflow, saves artefacts.

    Args:
        args: Parsed arguments (model_path, mlflow_uri)

    Returns:
        Dict of all computed metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("RadiologyAI — Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device     : {device}")
    logger.info(f"Model path : {args.model_path}")

    # ── Load model ────────────────────────────────────────────
    model = get_model(device, freeze=False)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )
    model.eval()
    logger.info("Model loaded successfully.")

    # ── Load test data ────────────────────────────────────────
    loaders = get_dataloaders(
        PROC_DIR,
        batch_size=get_config_value("ml", "evaluation", "batch_size", default=32),
    )

    # ── Get predictions ───────────────────────────────────────
    labels, preds, probs = get_predictions(model, loaders["test"], device)

    # ── Compute metrics ───────────────────────────────────────
    accuracy   = accuracy_score(labels, preds)
    macro_f1   = f1_score(labels, preds, average="macro",    zero_division=0)
    weighted_f1= f1_score(labels, preds, average="weighted", zero_division=0)
    macro_auc  = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    per_class_report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # sensitivity = recall per class
    # specificity = custom computation
    per_class_metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        binary = (labels == i).astype(int)
        _, _, auc_score = roc_curve(binary, probs[:, i])
        per_class_metrics[name] = {
            "precision"  : round(per_class_report[name]["precision"], 4),
            "recall"     : round(per_class_report[name]["recall"],    4),  # sensitivity
            "f1"         : round(per_class_report[name]["f1-score"],  4),
            "specificity": round(compute_specificity(labels, preds, i), 4),
            "auc"        : round(roc_auc_score(
                (labels == i).astype(int), probs[:, i]
            ), 4),
        }

    all_metrics = {
        "accuracy"    : round(float(accuracy),    4),
        "macro_f1"    : round(float(macro_f1),    4),
        "weighted_f1" : round(float(weighted_f1), 4),
        "macro_auc"   : round(float(macro_auc),   4),
        "per_class"   : per_class_metrics,
    }

    # ── Print summary ─────────────────────────────────────────
    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy    : {accuracy:.4f}")
    logger.info(f"  Macro F1    : {macro_f1:.4f}")
    logger.info(f"  Macro AUC   : {macro_auc:.4f}")
    logger.info(f"\nPer-class metrics:")
    for name, m in per_class_metrics.items():
        logger.info(
            f"  {name:<12} "
            f"precision={m['precision']:.3f}  "
            f"recall={m['recall']:.3f}  "
            f"specificity={m['specificity']:.3f}  "
            f"AUC={m['auc']:.3f}"
        )

    # ── Save artefacts ────────────────────────────────────────
    report_path = EXP_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nEval report saved: {report_path}")

    cm_path  = EXP_DIR / "confusion_matrix_test.png"
    roc_path = EXP_DIR / "roc_curves.png"
    save_confusion_matrix(labels, preds, cm_path)
    save_roc_curves(labels, probs, roc_path)

    # ── Log to MLflow ─────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "test_accuracy" : all_metrics["accuracy"],
            "test_macro_f1" : all_metrics["macro_f1"],
            "test_macro_auc": all_metrics["macro_auc"],
        })
        for name, m in per_class_metrics.items():
            mlflow.log_metrics({
                f"{name}_precision"  : m["precision"],
                f"{name}_recall"     : m["recall"],
                f"{name}_f1"         : m["f1"],
                f"{name}_specificity": m["specificity"],
                f"{name}_auc"        : m["auc"],
            })
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(report_path))

    # ── Check acceptance criteria ─────────────────────────────
    logger.info("\nAcceptance Criteria Check:")
    checks = {
        f"Accuracy > {ACCURACY_MIN:.2f}"   : accuracy   > ACCURACY_MIN,
        f"Macro F1 > {MACRO_F1_MIN:.2f}"   : macro_f1   > MACRO_F1_MIN,
        f"Macro AUC > {MACRO_AUC_MIN:.2f}" : macro_auc  > MACRO_AUC_MIN,
        f"Normal recall > {RECALL_MIN_PER_CLASS:.2f}"   : per_class_metrics["Normal"]["recall"]    > RECALL_MIN_PER_CLASS,
        f"Pneumonia recall > {RECALL_MIN_PER_CLASS:.2f}": per_class_metrics["Pneumonia"]["recall"] > RECALL_MIN_PER_CLASS,
        f"COVID19 recall > {RECALL_MIN_PER_CLASS:.2f}"  : per_class_metrics["COVID19"]["recall"]   > RECALL_MIN_PER_CLASS,
    }
    all_passed = True
    for criterion, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {criterion}")
        if not passed:
            all_passed = False

    logger.info(f"\nOverall: {'ALL CRITERIA MET' if all_passed else 'SOME CRITERIA NOT MET'}")
    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type    = str,
        default = str(resolve_path(get_config_value("paths", "model_path", default="ml/models/efficientnetb0_best.pth"))),
    )
    parser.add_argument(
        "--mlflow_uri",
        type    = str,
        default = get_config_value("ops", "mlflow", "service_url", default="http://localhost:5000"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
